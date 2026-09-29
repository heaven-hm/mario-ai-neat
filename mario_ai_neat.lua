-- Super Mario AI using NEAT plays Super Mario Bros. autonomously in FCEUX.
-- It learns controller decisions from RAM observations, fitness, and evolution.
-- Load this Lua 5.1 file in FCEUX with a compatible NES SMB1 ROM already open.

local AI = {}

-- Testing-only aid. Set this to false before a real, timed evaluation.
-- SMB1 stores each timer digit separately; keeping all three at 9 prevents a
-- training attempt from ending solely because the in-game clock expires.
local TESTING_FREEZE_TIMER = true
-- Testing-only aid. This keeps the SMB1 life counter replenished. It does not
-- revive Mario or skip the normal death and respawn sequence.
local TESTING_INFINITE_LIVES = true
-- FCEUX slot 9 is reserved for the AI's fixed training start. Named slots
-- reload safely without the persist() call that crashes some FCEUX builds.
local USE_FIXED_TRAINING_STATE = true
local TRAINING_SAVESTATE_SLOT = 9
-- Set true after training to replay the strongest saved genome only.
local PLAY_CHAMPION_ONLY = false
-- Click the small HUD tab in the upper-right corner to show/hide the live
-- network inspector. Drawing can be disabled here to maximize training speed.
local SHOW_NEURAL_INSPECTOR = true
local HUD_CLICK_COOLDOWN = 0

local RAM = {
  player_state=0x000E, enemy_present=0x000F, enemy_id=0x0016,
  enemy_state=0x001E, player_page=0x006D, enemy_page=0x006E,
  player_x=0x0086, enemy_x=0x0087, player_vx=0x0057,
  player_vy=0x009F, enemy_y=0x00CF, player_y=0x03B8,
  player_screen_x=0x03AD, death_music=0x0712,
  flag_event=0x010E, flag_y=0x070F, tiles=0x0500,
  player_size=0x0754, power=0x0756, operation_mode=0x0770,
  timer_hundreds=0x07F8, timer_tens=0x07F9, timer_ones=0x07FA,
  lives=0x075A,
}

local ENEMY_NAME = {
  [0x00]="green koopa", [0x02]="buzzy beetle", [0x03]="red koopa",
  [0x05]="hammer bro", [0x06]="goomba", [0x07]="bloober",
  [0x08]="bullet bill", [0x09]="paratroopa", [0x0A]="cheep-cheep",
  [0x0B]="cheep-cheep", [0x0C]="podoboo", [0x0D]="piranha plant",
  [0x0E]="jumping paratroopa", [0x0F]="red paratroopa",
  [0x10]="flying paratroopa", [0x11]="lakitu", [0x12]="spiny",
  [0x14]="flying cheep-cheep", [0x15]="bowser flame", [0x2D]="bowser",
  [0x33]="bullet bill",
}

local NON_SOLID_TILES = {[0x00]=true,[0x08]=true,[0x24]=true,[0x25]=true,[0x26]=true,
  [0x88]=true,[0xC2]=true,[0xC3]=true,[0xC5]=true}
local INACTIVE_ENEMY_STATES = {[0x02]=true,[0x03]=true,[0x04]=true,[0x20]=true,[0x22]=true,
  [0x23]=true,[0x83]=true,[0x84]=true,[0xC4]=true}

local function toSignedByte(value) return value >= 128 and value - 256 or value end
local function clamp(value, low, high) return math.max(low, math.min(high, value)) end

local function readByte(address) return memory.readbyte(address) end

-- Convert the SMB1 RAM layout into one readable snapshot for the learner.
function AI.observe(frameNumber)
  local marioWorldX = readByte(RAM.player_page) * 256 + readByte(RAM.player_x)
  local operationMode = readByte(RAM.operation_mode)
  local playerState = readByte(RAM.player_state)
  local state = {frame=frameNumber,worldX=marioWorldX,worldY=readByte(RAM.player_y)+16,
    screenX=readByte(RAM.player_screen_x),
    horizontalVelocity=toSignedByte(readByte(RAM.player_vx))/16,
    verticalVelocity=toSignedByte(readByte(RAM.player_vy)),playerState=playerState,
    size=readByte(RAM.player_size),power=readByte(RAM.power),operationMode=operationMode,
    enemies={},items={},tiles={},phase="playing"}

  if operationMode == 0 then
    state.phase = "title"
  elseif readByte(RAM.death_music) == 1 or playerState == 0x0B then
    state.phase = "death"
  elseif operationMode ~= 1 then
    state.phase = "transition"
  elseif playerState ~= 0x08 then
    state.phase = "locked"
  end
  if readByte(RAM.flag_event) == 0x3E and readByte(RAM.flag_y) == 0xA0 then
    state.phase = "victory"
  end

  -- The game keeps two 16-column pages of 16x16 metatiles in the tile buffer.
  -- This mapping must match the disassembly/ROM revision named in docs/ram-map.md.
  for tileAddressOffset = 0, 415 do
    state.tiles[tileAddressOffset] = readByte(RAM.tiles + tileAddressOffset)
  end
  for enemySlot = 0, 4 do
    if readByte(RAM.enemy_present + enemySlot) ~= 0 then
      local enemyId = readByte(RAM.enemy_id + enemySlot)
      state.enemies[#state.enemies+1] = {
        slot=enemySlot,
        id=enemyId,
        name=ENEMY_NAME[enemyId] or "unknown object",
        status=readByte(RAM.enemy_state + enemySlot),
        worldX=readByte(RAM.enemy_page + enemySlot)*256 + readByte(RAM.enemy_x + enemySlot),
        worldY=readByte(RAM.enemy_y + enemySlot)+24,
        horizontalVelocity=toSignedByte(readByte(0x0058 + enemySlot))/16,
      }
    end
  end
  if readByte(0x0014) == 1 and readByte(0x001B) == 0x2E then
    local itemWorldX = math.floor(marioWorldX/256)*256 + readByte(0x008C)
    if itemWorldX < marioWorldX-128 then
      itemWorldX = itemWorldX+256
    elseif itemWorldX > marioWorldX+128 then
      itemWorldX = itemWorldX-256
    end
    state.items[1] = {kind="powerup",type=readByte(0x0039),worldX=itemWorldX,
      worldY=readByte(0x03BE)+16,heading=readByte(0x004B)}
  end
  return state
end


-- The policy learns with NEAT-style neuroevolution. Each genome controls a
-- real SMB1 play segment; episode fitness selects parents for the next
-- generation. RAM writes are limited to the optional timer and lives test aids.
local SENSOR_RADIUS_TILES = 6
local GRID_WIDTH = SENSOR_RADIUS_TILES * 2 + 1
local GRID_INPUT_COUNT = GRID_WIDTH * GRID_WIDTH
local GLOBAL_INPUT_COUNT = 15
local OBSERVATION_INPUT_COUNT = GRID_INPUT_COUNT + GLOBAL_INPUT_COUNT
local NEURAL_INPUT_COUNT = OBSERVATION_INPUT_COUNT + 1 -- final input is the bias node
local ACTION_COUNT = 6
local OUTPUT_NODE_OFFSET = 1000000
local DEFAULT_POPULATION_SIZE = 300
local SPECIES_DISTANCE_THRESHOLD = 1.0
local MAX_STALE_GENERATIONS = 15
local SAVE_INTERVAL_FRAMES = 600

local ACTION_OPTIONS = {
  {name="run", right=true, B=true},
  {name="jump_run", right=true, B=true, A=true},
  {name="retreat", left=true, B=true},
  {name="brake"},
  {name="jump_place", A=true},
  {name="walk", right=true},
}
local NON_STOMPABLE_ENEMIES = {[0x07]=true,[0x0C]=true,[0x0D]=true,[0x11]=true,[0x12]=true}

local function safeWriteLine(file, value) file:write(value, "\n") end

function AI.inputCount() return NEURAL_INPUT_COUNT end
function AI.outputNode(index) return OUTPUT_NODE_OFFSET + index end
function AI.sensorIndex(horizontalOffset, verticalOffset)
  local columnIndex = math.floor((horizontalOffset + SENSOR_RADIUS_TILES * 16) / 16)
  local rowIndex = math.floor((verticalOffset + SENSOR_RADIUS_TILES * 16) / 16)
  return rowIndex * GRID_WIDTH + columnIndex + 1
end

local function isSolidTileAtOffset(state, horizontalOffset, verticalOffset)
  local sampledWorldX = state.worldX + horizontalOffset
  local sampledWorldY = state.worldY + verticalOffset - 16
  local columnIndex = math.floor((sampledWorldX+8)/16)
  local rowIndex = math.floor((sampledWorldY-32)/16)
  if rowIndex < 0 or rowIndex >= 13 then return false end
  local tileIndex = (math.floor(columnIndex/16)%2)*208 + rowIndex*16 + columnIndex%16
  local tileId = state.tiles[tileIndex]
  return tileId ~= nil and tileId ~= 0 and not NON_SOLID_TILES[tileId]
end

local function hasGapAhead(state)
  for horizontalOffset=16,64,16 do
    if not isSolidTileAtOffset(state,horizontalOffset,16) then return 1 end
  end
  return 0
end

local function encodeGridCell(state, horizontalOffset, verticalOffset)
  local sampledWorldX = state.worldX + horizontalOffset
  local sampledWorldY = state.worldY + verticalOffset - 16
  local columnIndex = math.floor((sampledWorldX + 8) / 16)
  local rowIndex = math.floor((sampledWorldY - 32) / 16)
  if rowIndex < 0 or rowIndex >= 13 then return 0 end
  local pageIndex = math.floor(columnIndex / 16) % 2
  local columnWithinPage = columnIndex % 16
  local tileId = state.tiles[pageIndex * 208 + rowIndex * 16 + columnWithinPage]
  local isOccupied = tileId ~= nil and tileId ~= 0 and not NON_SOLID_TILES[tileId]
  local encodedValue = isOccupied and 1 or 0
  for _, enemy in ipairs(state.enemies) do
    if not INACTIVE_ENEMY_STATES[enemy.status]
      and math.abs(enemy.worldX - (state.worldX + horizontalOffset)) <= 8
      and math.abs(enemy.worldY - (state.worldY + verticalOffset)) <= 8 then
      encodedValue = -1
      break
    end
  end
  return encodedValue
end

local function findNearestEnemy(state)
  local nearestEnemy, nearestDistance
  for _, enemy in ipairs(state.enemies) do
    if not INACTIVE_ENEMY_STATES[enemy.status] then
      local horizontalOffset, verticalOffset = enemy.worldX - state.worldX, enemy.worldY - state.worldY
      local weightedDistance = math.abs(horizontalOffset) + math.abs(verticalOffset) * 1.5
      if weightedDistance < 240 and (not nearestDistance or weightedDistance < nearestDistance) then
        nearestEnemy, nearestDistance = enemy, weightedDistance
      end
    end
  end
  return nearestEnemy, nearestDistance
end

-- Build the 184 observed values. AI.evaluateGenome adds the constant bias node.
function AI.buildObservationInputs(state)
  local inputValues = {}
  for verticalOffset=-SENSOR_RADIUS_TILES*16,SENSOR_RADIUS_TILES*16,16 do
    for horizontalOffset=-SENSOR_RADIUS_TILES*16,SENSOR_RADIUS_TILES*16,16 do
      inputValues[#inputValues+1] = encodeGridCell(state,horizontalOffset,verticalOffset)
    end
  end
  local nearestEnemy = findNearestEnemy(state)
  local enemyHorizontalOffset, enemyVerticalOffset = 0, 0
  local enemyHorizontalVelocity, enemyTypeValue = 0, 0
  if nearestEnemy then
    enemyHorizontalOffset = clamp((nearestEnemy.worldX-state.worldX)/128,-1,1)
    enemyVerticalOffset = clamp((nearestEnemy.worldY-state.worldY)/96,-1,1)
    enemyHorizontalVelocity = clamp(nearestEnemy.horizontalVelocity/4,-1,1)
    enemyTypeValue = (nearestEnemy.id or 0)/51
  end
  inputValues[#inputValues+1] = clamp(state.horizontalVelocity/4,-1,1)
  inputValues[#inputValues+1] = clamp(state.verticalVelocity/8,-1,1)
  inputValues[#inputValues+1] = state.grounded and 1 or -1
  inputValues[#inputValues+1] = state.size == 1 and -1 or 1
  inputValues[#inputValues+1] = state.power == 2 and 1 or (state.power == 1 and 0 or -1)
  inputValues[#inputValues+1] = enemyHorizontalOffset
  inputValues[#inputValues+1] = enemyVerticalOffset
  inputValues[#inputValues+1] = enemyHorizontalVelocity
  inputValues[#inputValues+1] = enemyTypeValue
  local visibleItem = state.items and state.items[1]
  if visibleItem then
    inputValues[#inputValues+1] = 1
    inputValues[#inputValues+1] = clamp((visibleItem.worldX-state.worldX)/160,-1,1)
    inputValues[#inputValues+1] = clamp((visibleItem.worldY-state.worldY)/96,-1,1)
    inputValues[#inputValues+1] = clamp((visibleItem.type or 0)/3,-1,1)
  else
    inputValues[#inputValues+1] = -1
    inputValues[#inputValues+1] = 0
    inputValues[#inputValues+1] = 0
    inputValues[#inputValues+1] = 0
  end
  inputValues[#inputValues+1] = hasGapAhead(state)
  inputValues[#inputValues+1] = nearestEnemy~=nil and nearestEnemy.worldX>state.worldX
    and nearestEnemy.worldX-state.worldX<32 and 1 or 0
  return inputValues
end

local function sigmoid(value)
  value = clamp(value,-60,60)
  return 2/(1+math.exp(-4.9*value))-1
end

-- Evaluate nodes in numeric order. Hidden node IDs grow upward and output IDs
-- begin at one million, so every feed-forward source is evaluated first.
function AI.evaluateGenome(genome, inputValues)
  local nodeValues, incomingConnections, activeNodes = {}, {}, {}
  for inputIndex = 1, OBSERVATION_INPUT_COUNT do
    nodeValues[inputIndex] = inputValues[inputIndex] or 0
  end
  nodeValues[NEURAL_INPUT_COUNT] = 1
  activeNodes[NEURAL_INPUT_COUNT] = true
  for actionIndex = 1, ACTION_COUNT do
    activeNodes[OUTPUT_NODE_OFFSET+actionIndex] = true
  end
  for _, gene in ipairs(genome.genes) do
    if gene.enabled then
      incomingConnections[gene.targetNode] = incomingConnections[gene.targetNode] or {}
      incomingConnections[gene.targetNode][#incomingConnections[gene.targetNode]+1] = gene
      activeNodes[gene.sourceNode], activeNodes[gene.targetNode] = true, true
    end
  end
  local evaluationOrder = {}
  for nodeId in pairs(activeNodes) do
    if nodeId > NEURAL_INPUT_COUNT then
      evaluationOrder[#evaluationOrder+1] = nodeId
    end
  end
  table.sort(evaluationOrder)
  for _, nodeId in ipairs(evaluationOrder) do
    local nodeConnections = incomingConnections[nodeId]
    if nodeConnections then
      local weightedSum = 0
      for _, gene in ipairs(nodeConnections) do
        weightedSum = weightedSum + (nodeValues[gene.sourceNode] or 0)*gene.weight
      end
      nodeValues[nodeId] = sigmoid(weightedSum)
    end
  end
  local actionScores = {}
  for actionIndex = 1, ACTION_COUNT do
    actionScores[actionIndex] = nodeValues[OUTPUT_NODE_OFFSET+actionIndex] or 0
  end
  return actionScores,nodeValues
end

local function createEmptyGenome()
  return {genes={},fitness=0,adjustedFitness=0,highestHiddenNode=NEURAL_INPUT_COUNT,
    mutationRates={connections=0.8,link=1.0,bias=0.4,node=0.12,enable=0.2,disable=0.2,step=0.1}}
end

local function connectionKey(sourceNode, targetNode)
  return tostring(sourceNode)..":"..tostring(targetNode)
end

-- Reuse one historical number for the same structural connection. This is the
-- NEAT mechanism that aligns homologous genes during crossover.
local function getInnovationNumber(populationState, sourceNode, targetNode)
  local connectionIdentifier = connectionKey(sourceNode,targetNode)
  if not populationState.innovations[connectionIdentifier] then
    populationState.nextInnovation = populationState.nextInnovation + 1
    populationState.innovations[connectionIdentifier] = populationState.nextInnovation
  end
  return populationState.innovations[connectionIdentifier]
end

local function createGene(sourceNode, targetNode, weight, innovationNumber)
  return {sourceNode=sourceNode,targetNode=targetNode,weight=weight,
    enabled=true,innovation=innovationNumber}
end

local function hasConnection(genome, sourceNode, targetNode)
  for _, gene in ipairs(genome.genes) do
    if gene.sourceNode==sourceNode and gene.targetNode==targetNode then return true end
  end
  return false
end

local function chooseRandomNode(genome, populationState, inputOnly)
  local candidates = {}
  for inputIndex = 1, NEURAL_INPUT_COUNT do
    candidates[#candidates+1] = inputIndex
  end
  if not inputOnly then
    for _, gene in ipairs(genome.genes) do
      if gene.sourceNode > NEURAL_INPUT_COUNT and gene.sourceNode < OUTPUT_NODE_OFFSET then candidates[#candidates+1]=gene.sourceNode end
      if gene.targetNode > NEURAL_INPUT_COUNT and gene.targetNode < OUTPUT_NODE_OFFSET then candidates[#candidates+1]=gene.targetNode end
    end
  end
  return candidates[math.random(#candidates)]
end

local function mutateAddConnection(genome, populationState, biasOnly)
  local sourceNode = biasOnly and NEURAL_INPUT_COUNT or chooseRandomNode(genome,populationState,false)
  local targetNode = OUTPUT_NODE_OFFSET + math.random(ACTION_COUNT)
  if not biasOnly then
    local hiddenNodes = {}
    for _, gene in ipairs(genome.genes) do
      if gene.sourceNode > NEURAL_INPUT_COUNT and gene.sourceNode < OUTPUT_NODE_OFFSET then
        hiddenNodes[#hiddenNodes+1] = gene.sourceNode
      end
      if gene.targetNode > NEURAL_INPUT_COUNT and gene.targetNode < OUTPUT_NODE_OFFSET then
        hiddenNodes[#hiddenNodes+1] = gene.targetNode
      end
    end
    if #hiddenNodes > 0 and math.random(2) == 1 then
      targetNode = hiddenNodes[math.random(#hiddenNodes)]
    end
  end
  if sourceNode >= targetNode or hasConnection(genome,sourceNode,targetNode) then return end
  table.insert(genome.genes,createGene(sourceNode,targetNode,math.random()*4-2,
    getInnovationNumber(populationState,sourceNode,targetNode)))
end

local function mutateAddNode(genome, populationState)
  local enabledGenes = {}
  for _, gene in ipairs(genome.genes) do
    if gene.enabled then enabledGenes[#enabledGenes+1] = gene end
  end
  if #enabledGenes == 0 then return end
  local splitGene = enabledGenes[math.random(#enabledGenes)]
  splitGene.enabled=false
  genome.highestHiddenNode=math.max(genome.highestHiddenNode,NEURAL_INPUT_COUNT)+1
  if genome.highestHiddenNode>=OUTPUT_NODE_OFFSET then return end
  local newHiddenNode = genome.highestHiddenNode
  table.insert(genome.genes,createGene(splitGene.sourceNode,newHiddenNode,1,
    getInnovationNumber(populationState,splitGene.sourceNode,newHiddenNode)))
  table.insert(genome.genes,createGene(newHiddenNode,splitGene.targetNode,splitGene.weight,
    getInnovationNumber(populationState,newHiddenNode,splitGene.targetNode)))
end

local function repeatMutationByRate(rate, callback)
  while rate > 0 do
    if math.random() < math.min(1,rate) then callback() end
    rate = rate - 1
  end
end

function AI.mutate(genome, populationState)
  for mutationName, mutationRate in pairs(genome.mutationRates) do
    if mutationName ~= "step" then
      genome.mutationRates[mutationName] = mutationRate
        * (math.random(2) == 1 and 0.95 or 1.05263)
    end
  end
  if math.random() < genome.mutationRates.connections then
    for _, gene in ipairs(genome.genes) do
      if math.random() < 0.9 then
        gene.weight = gene.weight + (math.random()*2-1)*genome.mutationRates.step
      else
        gene.weight = math.random()*4-2
      end
    end
  end
  repeatMutationByRate(genome.mutationRates.link,function() mutateAddConnection(genome,populationState,false) end)
  repeatMutationByRate(genome.mutationRates.bias,function() mutateAddConnection(genome,populationState,true) end)
  repeatMutationByRate(genome.mutationRates.node,function() mutateAddNode(genome,populationState) end)
  repeatMutationByRate(genome.mutationRates.enable,function()
    local disabledGenes = {}
    for _, gene in ipairs(genome.genes) do
      if not gene.enabled then disabledGenes[#disabledGenes+1] = gene end
    end
    if #disabledGenes > 0 then disabledGenes[math.random(#disabledGenes)].enabled=true end
  end)
  repeatMutationByRate(genome.mutationRates.disable,function()
    local enabledGenes = {}
    for _, gene in ipairs(genome.genes) do
      if gene.enabled then enabledGenes[#enabledGenes+1] = gene end
    end
    if #enabledGenes > 0 then enabledGenes[math.random(#enabledGenes)].enabled=false end
  end)
end

local function cloneGenome(genome)
  local genomeCopy={genes={},fitness=genome.fitness or 0,adjustedFitness=0,
    highestHiddenNode=genome.highestHiddenNode,mutationRates={}}
  for mutationName, mutationRate in pairs(genome.mutationRates) do
    genomeCopy.mutationRates[mutationName] = mutationRate
  end
  for _, gene in ipairs(genome.genes) do
    genomeCopy.genes[#genomeCopy.genes+1]={sourceNode=gene.sourceNode,targetNode=gene.targetNode,weight=gene.weight,
      enabled=gene.enabled,innovation=gene.innovation}
  end
  return genomeCopy
end

function AI.newGenome(populationState)
  local genome=createEmptyGenome()
  -- Sparse initial networks keep the first generation diverse and inexpensive.
  for actionIndex = 1, ACTION_COUNT do
    local sourceNode = math.random(NEURAL_INPUT_COUNT)
    local targetNode = OUTPUT_NODE_OFFSET + actionIndex
    table.insert(genome.genes,createGene(sourceNode,targetNode,
      (math.random()*2-1)*0.5,getInnovationNumber(populationState,sourceNode,targetNode)))
  end
  return genome
end

local function createSeededGenome(populationState)
  local genome=createEmptyGenome()
  local enemyHorizontalOffsetInput = GRID_INPUT_COUNT+6
  local itemHorizontalOffsetInput = GRID_INPUT_COUNT+11
  local gapAheadInput = GRID_INPUT_COUNT+14
  local enemyContactInput = GRID_INPUT_COUNT+15
  local function connect(sourceNode, actionIndex, weight)
    local targetNode = OUTPUT_NODE_OFFSET + actionIndex
    table.insert(genome.genes,createGene(sourceNode,targetNode,weight,
      getInnovationNumber(populationState,sourceNode,targetNode)))
  end
  -- Start from sensible SMB1 play: run on clear ground, jump for an enemy or
  -- pit, and turn back toward a visible reward. Evolution can change all links.
  connect(NEURAL_INPUT_COUNT,1,0.55)
  connect(NEURAL_INPUT_COUNT,2,-0.28)
  connect(enemyHorizontalOffsetInput,2,3.4)
  connect(gapAheadInput,2,3.0)
  connect(itemHorizontalOffsetInput,3,-1.2)
  connect(enemyHorizontalOffsetInput,3,-2.0)
  connect(NEURAL_INPUT_COUNT,5,-0.2)
  connect(enemyContactInput,5,1.5)
  return genome
end

local function calculateGenomeDistance(firstGenome, secondGenome)
  local secondGenesByInnovation = {}
  for _, gene in ipairs(secondGenome.genes) do
    secondGenesByInnovation[gene.innovation] = gene
  end
  local matchingGeneCount, totalWeightDifference, unmatchedGeneCount = 0, 0, 0
  for _, gene in ipairs(firstGenome.genes) do
    local matchingGene = secondGenesByInnovation[gene.innovation]
    if matchingGene then
      matchingGeneCount = matchingGeneCount+1
      totalWeightDifference = totalWeightDifference+math.abs(gene.weight-matchingGene.weight)
    else
      unmatchedGeneCount = unmatchedGeneCount+1
    end
  end
  unmatchedGeneCount = unmatchedGeneCount
    + math.max(0,#secondGenome.genes-matchingGeneCount)
  local genomeSize = math.max(1,#firstGenome.genes,#secondGenome.genes)
  local averageWeightDifference = matchingGeneCount > 0
    and totalWeightDifference/matchingGeneCount or 0
  return 2*unmatchedGeneCount/genomeSize + 0.4*averageWeightDifference
end

local function assignSpecies(populationState)
  local previousSpecies = populationState.species or {}
  local currentSpecies = {}
  for _, genome in ipairs(populationState.genomes) do
    local matchingSpecies
    for _, speciesGroup in ipairs(currentSpecies) do
      if calculateGenomeDistance(genome,speciesGroup.representative)<SPECIES_DISTANCE_THRESHOLD then
        matchingSpecies=speciesGroup
        break
      end
    end
    if not matchingSpecies then
      local previousMatch
      for _, speciesGroup in ipairs(previousSpecies) do
        if calculateGenomeDistance(genome,speciesGroup.representative)<SPECIES_DISTANCE_THRESHOLD then
          previousMatch=speciesGroup
          break
        end
      end
      matchingSpecies={id=previousMatch and previousMatch.id or (#currentSpecies+1),genomes={},
        topFitness=previousMatch and previousMatch.topFitness or 0,
        staleness=previousMatch and previousMatch.staleness or 0,
        representative=cloneGenome(genome)}
      currentSpecies[#currentSpecies+1]=matchingSpecies
    end
    matchingSpecies.genomes[#matchingSpecies.genomes+1]=genome
    genome.species=matchingSpecies.id
  end
  populationState.species=currentSpecies
end

-- Create one seeded SMB1 controller, then mutate clones until the requested
-- population size is reached. The first genome provides a useful prior; NEAT
-- is free to replace that prior during later generations.
function AI.newPopulation(populationSize)
  local populationState = {generation=1,nextInnovation=ACTION_COUNT,innovations={},genomes={},species={},
    bestFitness=0,population=populationSize or DEFAULT_POPULATION_SIZE}
  for genomeIndex = 1, populationState.population do
    local genome
    if genomeIndex == 1 then
      genome = createSeededGenome(populationState)
    else
      genome = cloneGenome(populationState.genomes[1])
      AI.mutate(genome,populationState)
    end
    populationState.genomes[#populationState.genomes+1]=genome
  end
  assignSpecies(populationState)
  return populationState
end

local function crossover(first,second)
  if second.fitness > first.fitness then first, second = second, first end
  local child = createEmptyGenome()
  local secondGenesByInnovation = {}
  for _, gene in ipairs(second.genes) do
    secondGenesByInnovation[gene.innovation] = gene
  end
  for _, firstGene in ipairs(first.genes) do
    local matchingGene = secondGenesByInnovation[firstGene.innovation]
    local selectedGene = matchingGene and math.random(2) == 1 and matchingGene or firstGene
    local childGene = {sourceNode=selectedGene.sourceNode,targetNode=selectedGene.targetNode,
      weight=selectedGene.weight,enabled=selectedGene.enabled,innovation=selectedGene.innovation}
    if matchingGene and ((not firstGene.enabled) or (not matchingGene.enabled)) and math.random() < 0.75 then
      childGene.enabled = false
    end
    child.genes[#child.genes+1] = childGene
  end
  child.highestHiddenNode = math.max(first.highestHiddenNode,second.highestHiddenNode)
  for mutationName, mutationRate in pairs(first.mutationRates) do
    child.mutationRates[mutationName] = mutationRate
  end
  return child
end

local function rankSpecies(populationState)
  table.sort(populationState.genomes,function(firstGenome,secondGenome)
    return firstGenome.fitness > secondGenome.fitness
  end)
  for rank, genome in ipairs(populationState.genomes) do
    genome.globalRank = #populationState.genomes-rank+1
  end
  local champion = populationState.genomes[1]
  populationState.bestFitness = math.max(populationState.bestFitness or 0,
    champion and champion.fitness or 0)
  for _, speciesGroup in ipairs(populationState.species) do
    table.sort(speciesGroup.genomes,function(firstGenome,secondGenome)
      return firstGenome.fitness > secondGenome.fitness
    end)
    local topFitness = speciesGroup.genomes[1] and speciesGroup.genomes[1].fitness or 0
    if topFitness > speciesGroup.topFitness then
      speciesGroup.topFitness = topFitness
      speciesGroup.staleness = 0
    else
      speciesGroup.staleness = (speciesGroup.staleness or 0)+1
    end
    local adjustedFitnessTotal = 0
    for _, genome in ipairs(speciesGroup.genomes) do
      genome.adjustedFitness = genome.globalRank/math.max(1,#speciesGroup.genomes)
      adjustedFitnessTotal = adjustedFitnessTotal+genome.adjustedFitness
    end
    speciesGroup.averageFitness = adjustedFitnessTotal/math.max(1,#speciesGroup.genomes)
  end
end

local function chooseSpeciesForBreeding(species)
  local totalFitness = 0
  for _, speciesGroup in ipairs(species) do
    totalFitness = totalFitness + math.max(0,speciesGroup.averageFitness or 0)
  end
  if totalFitness <= 0 then return species[math.random(#species)] end
  local selectionPoint = math.random()*totalFitness
  for _, speciesGroup in ipairs(species) do
    selectionPoint = selectionPoint-math.max(0,speciesGroup.averageFitness or 0)
    if selectionPoint <= 0 then return speciesGroup end
  end
  return species[#species]
end

local function breedChild(group,populationState)
  local speciesMembers = group.genomes
  if #speciesMembers == 1 then
    local child = cloneGenome(speciesMembers[1])
    child.fitness, child.adjustedFitness = 0, 0
    return child
  end
  local firstParent = speciesMembers[math.random(#speciesMembers)]
  local secondParent = speciesMembers[math.random(#speciesMembers)]
  local child = math.random() < 0.75
    and crossover(firstParent,secondParent) or cloneGenome(firstParent)
  child.fitness, child.adjustedFitness = 0, 0
  AI.mutate(child,populationState)
  return child
end

function AI.nextGeneration(populationState)
  rankSpecies(populationState)
  local champion = populationState.genomes[1]
  local survivingSpecies = {}
  for _, speciesGroup in ipairs(populationState.species) do
    if speciesGroup.staleness < MAX_STALE_GENERATIONS or speciesGroup.genomes[1] == champion then
      local survivorCount = math.max(1,math.ceil(#speciesGroup.genomes/2))
      for memberIndex = #speciesGroup.genomes, survivorCount+1, -1 do
        speciesGroup.genomes[memberIndex] = nil
      end
      survivingSpecies[#survivingSpecies+1] = speciesGroup
    end
  end
  if #survivingSpecies == 0 then
    survivingSpecies = {{id=1,genomes={champion},topFitness=champion.fitness,staleness=0,
      representative=cloneGenome(champion),averageFitness=1}}
  end
  local nextPopulation = {generation=populationState.generation+1,
    nextInnovation=populationState.nextInnovation,innovations=populationState.innovations,
    species=survivingSpecies,genomes={cloneGenome(champion)},
    bestFitness=populationState.bestFitness,population=populationState.population}
  local targetPopulationSize = populationState.population or DEFAULT_POPULATION_SIZE
  while #nextPopulation.genomes < targetPopulationSize do
    local speciesGroup = chooseSpeciesForBreeding(survivingSpecies)
    nextPopulation.genomes[#nextPopulation.genomes+1] = breedChild(speciesGroup,nextPopulation)
  end
  assignSpecies(nextPopulation)
  return nextPopulation
end

local function getDatabasePath()
  local source=debug and debug.getinfo and debug.getinfo(1,"S").source or ""
  if source:sub(1,1)=="@" then
    local script=source:sub(2)
    local folder=script:match("^(.*[/\\])") or ""
    return folder.."mario_ai_neat.db"
  end
  return "mario_ai_neat.db"
end

local function getLogPath()
  local source=debug and debug.getinfo and debug.getinfo(1,"S").source or ""
  if source:sub(1,1)=="@" then
    local script=source:sub(2)
    local folder=script:match("^(.*[/\\])") or ""
    return folder.."mario_ai_neat.log"
  end
  return "mario_ai_neat.log"
end

function AI.appendLog(message,path)
  path=path or getLogPath()
  if not io or not io.open then return false end
  local file=io.open(path,"a")
  if not file then return false end
  local stamp=os and os.date and os.date("%Y-%m-%d %H:%M:%S") or "time-unknown"
  file:write("[",stamp,"] ",tostring(message),"\n")
  file:flush()
  file:close()
  return true
end

-- object(slot) is the current FCEUX API. create(slot) is retained by FCEUX
-- for older builds and uses an offset slot number. Never call persist().
function AI.createStateAdapter(api,slot)
  if type(api)~="table" or type(api.save)~="function" or type(api.load)~="function" then
    return nil,"savestate API unavailable"
  end
  local stateConstructor, constructorArgument, adapterKind
  if type(api.object)=="function" then
    stateConstructor, constructorArgument, adapterKind = api.object, slot, "object"
  elseif type(api.create)=="function" then
    stateConstructor, constructorArgument, adapterKind = api.create, slot+1, "create"
  else return nil,"savestate constructor unavailable" end
  local succeeded, stateHandle = pcall(stateConstructor,constructorArgument)
  if not succeeded or stateHandle == nil then return nil,"could not create slot handle" end
  local adapter={handle=stateHandle,slot=slot,kind=adapterKind}
  function adapter:save()
    local succeeded, result = pcall(api.save,self.handle)
    return succeeded and result ~= false
  end
  function adapter:load()
    local succeeded, result = pcall(api.load,self.handle)
    return succeeded and result ~= false
  end
  return adapter
end

function AI.freezeTimerForTesting()
  if not TESTING_FREEZE_TIMER or not memory or not memory.writebyte then return false end
  memory.writebyte(RAM.timer_hundreds,0x09)
  memory.writebyte(RAM.timer_tens,0x09)
  memory.writebyte(RAM.timer_ones,0x09)
  return true
end

function AI.keepLivesForTesting()
  if not TESTING_INFINITE_LIVES or not memory or not memory.writebyte then return false end
  -- The inherited LuaRio test section identifies 0x075A as the lives byte.
  -- Refreshing 9 gives the practical effect of infinite lives during tests.
  memory.writebyte(RAM.lives,0x09)
  return true
end

function AI.save(populationState,path)
  -- Write a complete temporary checkpoint first. The rename keeps the last
  -- valid population available if FCEUX stops while the file is being written.
  path = path or getDatabasePath()
  local temporaryPath = path..".tmp"
  local databaseFile = io and io.open and io.open(temporaryPath,"w")
  if not databaseFile then return false end
  safeWriteLine(databaseFile,table.concat({"MARIO_AI_NEAT_V1",populationState.generation,populationState.nextInnovation,
    populationState.bestFitness or 0,populationState.population or #populationState.genomes,#populationState.genomes},","))
  for genomeIndex, genome in ipairs(populationState.genomes) do
    safeWriteLine(databaseFile,table.concat({"G",genomeIndex,genome.fitness or 0,
      genome.highestHiddenNode or NEURAL_INPUT_COUNT,genome.species or 0},","))
    for mutationName, mutationRate in pairs(genome.mutationRates) do
      safeWriteLine(databaseFile,table.concat({"R",genomeIndex,mutationName,mutationRate},","))
    end
    for _, gene in ipairs(genome.genes) do
      safeWriteLine(databaseFile,table.concat({"N",genomeIndex,gene.sourceNode,gene.targetNode,string.format("%.17g",gene.weight),
        gene.enabled and 1 or 0,gene.innovation},","))
    end
  end
  databaseFile:close()
  local renamed = os and os.rename and os.rename(temporaryPath,path)
  if not renamed then
    local temporaryFile = io.open(temporaryPath,"r")
    local destinationFile = io.open(path,"w")
    if not temporaryFile or not destinationFile then
      if temporaryFile then temporaryFile:close() end
      if destinationFile then destinationFile:close() end
      return false
    end
    destinationFile:write(temporaryFile:read("*a"))
    temporaryFile:close()
    destinationFile:close()
    os.remove(temporaryPath)
  end
  return true
end

function AI.load(path)
  -- Loading is deliberately tolerant of missing or malformed checkpoints: a
  -- failed load returns nil and AI.run creates a fresh population instead.
  path = path or getDatabasePath()
  if not io or not io.open then return nil end
  local databaseFile = io.open(path,"r")
  if not databaseFile then return nil end
  local header = databaseFile:read("*l") or ""
  local generation, innovationId, bestFitness, populationSize, genomeCount = header:match(
    "^MARIO_AI_NEAT_V1,(%d+),([%d%.]+),([%d%.%-]+),(%d+),(%d+)$")
  if not generation then databaseFile:close();return nil end
  local populationState={generation=tonumber(generation),nextInnovation=tonumber(innovationId),
    bestFitness=tonumber(bestFitness),population=tonumber(populationSize),genomes={},species={},innovations={}}
  for genomeIndex = 1, tonumber(genomeCount) do
    populationState.genomes[genomeIndex] = createEmptyGenome()
    populationState.genomes[genomeIndex].fitness = 0
  end
  for line in databaseFile:lines() do
    local fields={}
    for field in (line..","):gmatch("(.-),") do fields[#fields+1]=field end
    if fields[1]=="G" then
      local genomeIndex = tonumber(fields[2])
      if populationState.genomes[genomeIndex] then
        populationState.genomes[genomeIndex].fitness=tonumber(fields[3]) or 0
        populationState.genomes[genomeIndex].highestHiddenNode=tonumber(fields[4]) or NEURAL_INPUT_COUNT
        populationState.genomes[genomeIndex].species=tonumber(fields[5]) or 0
      end
    elseif fields[1]=="R" then
      local genomeIndex = tonumber(fields[2])
      if populationState.genomes[genomeIndex] then
        populationState.genomes[genomeIndex].mutationRates[fields[3]]=tonumber(fields[4]) or 0
      end
    elseif fields[1]=="N" then
      local genomeIndex = tonumber(fields[2])
      if populationState.genomes[genomeIndex] then
        local gene={sourceNode=tonumber(fields[3]),targetNode=tonumber(fields[4]),weight=tonumber(fields[5]),
          enabled=tonumber(fields[6])==1,innovation=tonumber(fields[7])}
        populationState.genomes[genomeIndex].genes[#populationState.genomes[genomeIndex].genes+1]=gene
        populationState.innovations[connectionKey(gene.sourceNode,gene.targetNode)]=gene.innovation
      end
    end
  end
  databaseFile:close()
  assignSpecies(populationState)
  return populationState
end

local function findClosestThreat(state)
  local nearestEnemy, nearestDistance
  for _, enemyCandidate in ipairs(state.enemies) do
    local horizontalOffset = enemyCandidate.worldX-state.worldX
    local verticalOffset = math.abs(enemyCandidate.worldY-state.worldY)
    if not INACTIVE_ENEMY_STATES[enemyCandidate.status] and verticalOffset<72
      and horizontalOffset>-32 and horizontalOffset<144 then
      local weightedDistance = math.abs(horizontalOffset)+verticalOffset
      if not nearestDistance or weightedDistance<nearestDistance then
        nearestEnemy, nearestDistance = enemyCandidate, weightedDistance
      end
    end
  end
  return nearestEnemy
end

local function calculateAllowedActions(state,enemy)
  -- This is a narrow safety shield, not a second controller. It removes only
  -- actions that would make an observed close threat immediately unavoidable.
  local allowed={}
  if enemy then
    local horizontalOffset=enemy.worldX-state.worldX
    if horizontalOffset>0 and horizontalOffset<112 then
      -- Preserve the original AI's stomp response for ground enemies. Close
      -- contact and non-stompable enemies remove forward motion from the policy.
      allowed[3],allowed[4]=true,true
    if not NON_STOMPABLE_ENEMIES[enemy.id] and horizontalOffset>=28 then
        allowed[2]=true
        if state.grounded then allowed[5]=true end
      elseif state.grounded then
        allowed[5]=true
      end
      if state.power==2 and horizontalOffset>36 and horizontalOffset>=28 then allowed[1]=true end
      return allowed
    elseif horizontalOffset<=0 and horizontalOffset>-32 then
      allowed[1],allowed[2],allowed[3],allowed[4],allowed[5]=true,true,true,true,true
      return allowed
    end
  end
  for actionIndex = 1, ACTION_COUNT do allowed[actionIndex] = true end
  return allowed
end

local function chooseAction(genome,state)
  local enemy=findClosestThreat(state)
  local allowed=calculateAllowedActions(state,enemy)
  local observationInputs=AI.buildObservationInputs(state)
  local actionScores,nodeValues=AI.evaluateGenome(genome,observationInputs)
  if state.power==2 and enemy and enemy.worldX-state.worldX>36 then actionScores[1]=actionScores[1]+0.3 end
  local selectedActionIndex,highestActionScore
  for actionIndex=1,ACTION_COUNT do
    if allowed[actionIndex] and (not highestActionScore or actionScores[actionIndex]>highestActionScore) then
      selectedActionIndex,highestActionScore=actionIndex,actionScores[actionIndex]
    end
  end
  return ACTION_OPTIONS[selectedActionIndex or 4],selectedActionIndex or 4,enemy,actionScores,nodeValues,observationInputs
end

local ACTION_LABEL = {
  run="run with speed", jump_run="running jump", retreat="retreat and reassess",
  brake="brake for control", jump_place="vertical jump", walk="controlled walk",
}

function AI.learningStatus(aiState,state,action)
  local enemy=findClosestThreat(state)
  local gapAhead=hasGapAhead(state)>0
  local visibleItem=state.items and state.items[1]
  local lesson="forward movement and momentum control"
  local sensing="clear ground ahead"
  if enemy then
    sensing=string.format("%s %d px ahead",enemy.name,math.max(0,enemy.worldX-state.worldX))
    if action.name=="jump_run" or action.name=="jump_place" then
      lesson="jump timing to clear an enemy"
    elseif action.name=="retreat" or action.name=="brake" then
      lesson="safe spacing and enemy avoidance"
    else
      lesson="choosing a safe response to an enemy"
    end
  elseif gapAhead then
    sensing="gap or missing floor ahead"
    lesson=action.name=="jump_run" and "jump timing and landing distance" or "safe gap approach"
  elseif visibleItem then
    sensing="power-up visible"
    lesson=action.name=="retreat" and "positioning for a power-up" or "power-up value versus safe progress"
  elseif not state.grounded then
    sensing="Mario is airborne"
    lesson="air control and landing alignment"
  end
  return {lesson=lesson,sensing=sensing,action=ACTION_LABEL[action.name] or action.name,
    progress=math.max(0,(aiState.furthestWorldX or state.worldX)-(aiState.startWorldX or state.worldX))}
end

function AI.new(populationState)
  return {populationState=populationState or AI.newPopulation(),genomeIndex=1,episodeFrames=0,
    startWorldX=nil,furthestWorldX=nil,lastProgressFrame=0,episodeReward=0,totalEpisodes=0,
    lastAction=nil,frames=0,finished=false,episodeActive=false,championMode=false}
end

function AI.bestGenomeIndex(populationState)
  local championIndex,bestFitness=1,nil
  for genomeIndex,genome in ipairs(populationState.genomes) do
    if bestFitness==nil or (genome.fitness or 0)>bestFitness then
      championIndex,bestFitness=genomeIndex,genome.fitness or 0
    end
  end
  return championIndex,bestFitness or 0
end

function AI.beginEpisode(aiState,state)
  aiState.episodeFrames=0;aiState.episodeReward=0;aiState.startWorldX=state.worldX;aiState.furthestWorldX=state.worldX
  aiState.lastProgressFrame=0;aiState.finished=false
  aiState.episodeActive=true
  aiState.bestForm=state.power==2 and 2 or (state.size==1 and 0 or 1)
end

function AI.decide(aiState,state)
  aiState.frames=aiState.frames+1
  if state.phase~="playing" then
    return {reason=state.phase}
  end
  if aiState.startWorldX==nil then AI.beginEpisode(aiState,state) end
  aiState.episodeFrames=aiState.episodeFrames+1
  if state.worldX>aiState.furthestWorldX then aiState.furthestWorldX=state.worldX;aiState.lastProgressFrame=aiState.episodeFrames end
  aiState.bestForm=math.max(aiState.bestForm or 0,state.power==2 and 2 or (state.size==1 and 0 or 1))
  local genome=aiState.populationState.genomes[aiState.genomeIndex]
  local action,actionIndex,enemy,actionScores,nodeValues,observationInputs=chooseAction(genome,state)
  action.name=ACTION_OPTIONS[actionIndex].name
  action.reason=enemy and ("learned "..action.name.." | threat "..enemy.name)
    or ("learned "..action.name)
  aiState.lastAction=actionIndex
  aiState.lastActionScores=actionScores
  aiState.lastNodeValues=nodeValues
  aiState.lastObservationInputs=observationInputs
  aiState.lastState=state
  return action
end

-- FCEUX uses ARGB colors for gui.drawbox/drawline. MarI/O's port converts to
-- the byte order expected by FCEUX; keep that conversion local to the HUD.
local function hudColor(argb)
  local alpha=math.floor(argb/0x1000000)%256
  local red=math.floor(argb/0x10000)%256
  local green=math.floor(argb/0x100)%256
  local blue=argb%256
  return alpha+blue*0x100+green*0x10000+red*0x1000000
end

local function hudText(guiApi,x,y,value,foreground,background)
  local text=tostring(value)
  local palette={white=0xFFFFFFFF,cyan=0xFF00FFFF,yellow=0xFFFFFF00,green=0xFF42FF70,
    red=0xFFFF5050,gray=0xFF9AA7B2,black=0xFF101010}
  if guiApi.drawtext then
    guiApi.drawtext(x,y,text,hudColor(palette[foreground] or 0xFFFFFFFF),0)
  elseif guiApi.text then
    guiApi.text(x,y,text,foreground or "white",background or "black")
  end
end

local function hudBox(guiApi,x1,y1,x2,y2,fill,outline)
  local draw=guiApi.drawbox or guiApi.box
  if draw then draw(x1,y1,x2,y2,hudColor(fill),hudColor(outline or fill)) end
end

local function hudLine(guiApi,x1,y1,x2,y2,color)
  local draw=guiApi.drawline or guiApi.line
  if draw then draw(x1,y1,x2,y2,hudColor(color)) end
end

local function makeHudNodePositions(genome)
  local positions={}
  local gridStartX,gridStartY,gridStep=4,50,2
  for inputIndex=1,GRID_INPUT_COUNT do
    local cellIndex=inputIndex-1
    local row=math.floor(cellIndex/GRID_WIDTH)
    local column=cellIndex%GRID_WIDTH
    positions[inputIndex]={x=gridStartX+column*gridStep,y=gridStartY+row*gridStep}
  end
  for featureIndex=1,GLOBAL_INPUT_COUNT do
    local inputIndex=GRID_INPUT_COUNT+featureIndex
    positions[inputIndex]={x=35,y=50+(featureIndex-1)*3}
  end
  positions[NEURAL_INPUT_COUNT]={x=35,y=99}

  local hiddenNodes={}
  for _,gene in ipairs(genome.genes or {}) do
    if gene.enabled then
      for _,nodeId in ipairs({gene.sourceNode,gene.targetNode}) do
        if nodeId>NEURAL_INPUT_COUNT and nodeId<OUTPUT_NODE_OFFSET and not positions[nodeId] then
          positions[nodeId]={pending=true}
          hiddenNodes[#hiddenNodes+1]=nodeId
        end
      end
    end
  end
  table.sort(hiddenNodes)
  for hiddenIndex,nodeId in ipairs(hiddenNodes) do
    local column=math.floor((hiddenIndex-1)/12)
    local row=(hiddenIndex-1)%12
    if hiddenIndex<=48 then positions[nodeId]={x=64+column*6,y=50+row*4} end
  end
  for actionIndex=1,ACTION_COUNT do
    positions[OUTPUT_NODE_OFFSET+actionIndex]={x=96,y=50+(actionIndex-1)*8}
  end
  return positions,#hiddenNodes
end

-- Draw a compact live inspector in the upper-left, leaving the game view clear.
function AI.drawNeuralInspector(guiApi,aiState,state,action,buttons)
  if not guiApi or not (guiApi.text or guiApi.drawtext) then return false end
  if not SHOW_NEURAL_INSPECTOR then
    hudText(guiApi,224,12,"[AI]","cyan","black")
    return false
  end
  local genome=aiState.populationState.genomes[aiState.genomeIndex]
  if not genome then return false end
  local nodePositions=makeHudNodePositions(genome)
  -- Three small panels occupy the upper-left corner only. The rest of the
  -- 256x240 game picture stays visible, including Mario near ground level.
  hudBox(guiApi,0,10,135,38,0xB0000000,0xB0000000)
  hudBox(guiApi,0,40,135,106,0x90000000,0x90000000)
  hudBox(guiApi,0,108,135,125,0xB0000000,0xB0000000)
  hudText(guiApi,2,12,"MARIO AI  NEAT","cyan","black")
  hudText(guiApi,2,20,string.format("G%d #%d/%d S%d",
    aiState.populationState.generation,aiState.genomeIndex,#aiState.populationState.genomes,
    genome.species or 0),"white","black")
  local progress=math.max(0,(aiState.furthestWorldX or state.worldX)-(aiState.startWorldX or state.worldX))
  local actionName=action and action.name or "idle"
  local shortAction=({run="RUN",jump_run="JUMP",retreat="BACK",brake="STOP",
    jump_place="HOP",walk="WALK"})[actionName] or "IDLE"
  hudText(guiApi,2,28,string.format("X%d +%d %s",state.worldX,progress,
    shortAction),"white","black")

  -- Keep the network rendering lightweight so the game remains responsive.
  local drawnConnections=0
  for _,gene in ipairs(genome.genes or {}) do
    if gene.enabled then
      local source,target=nodePositions[gene.sourceNode],nodePositions[gene.targetNode]
      if source and target and not source.pending and not target.pending and drawnConnections<50 then
        local color=gene.weight>=0 and 0x6000DD55 or 0x60FF453A
        hudLine(guiApi,source.x,source.y,target.x,target.y,color)
        drawnConnections=drawnConnections+1
      end
    end
  end

  -- Grid is the local tile view: dark empty, green solid, red enemy.
  hudText(guiApi,2,41,"INPUTS","white","black")
  for inputIndex=1,GRID_INPUT_COUNT do
    local point=nodePositions[inputIndex]
    local value=(aiState.lastObservationInputs or {})[inputIndex] or 0
    local color=value<0 and 0xFFFF5C64 or (value>0 and 0xFF56D69A or 0xFF344454)
    hudBox(guiApi,point.x,point.y,point.x+1,point.y+1,color,color)
  end

  -- Global features are a small activation column beside the 13x13 grid.
  for featureIndex=1,GLOBAL_INPUT_COUNT do
    local inputIndex=GRID_INPUT_COUNT+featureIndex
    local point=nodePositions[inputIndex]
    local value=(aiState.lastObservationInputs or {})[inputIndex] or 0
    local color=value<0 and 0xFFFF6873 or (value>0 and 0xFF62D6A5 or 0xFF617180)
    hudBox(guiApi,point.x,point.y,point.x+1,point.y+1,color,color)
  end
  local biasPoint=nodePositions[NEURAL_INPUT_COUNT]
  hudBox(guiApi,biasPoint.x,biasPoint.y,biasPoint.x+2,biasPoint.y+2,0xFFFFFF00,0xFFFFFF00)

  -- Hidden activations and output nodes are colored by their current value.
  for nodeId,point in pairs(nodePositions) do
    if nodeId>NEURAL_INPUT_COUNT and not point.pending then
      local activation=(aiState.lastNodeValues or {})[nodeId] or 0
      local color=activation>=0 and 0xFF62D6A5 or 0xFFFF6873
      hudBox(guiApi,point.x-1,point.y-1,point.x+1,point.y+1,color,color)
    end
  end
  local outputLabels={"run","jrun","back","stop","hop","walk"}
  for actionIndex,option in ipairs(ACTION_OPTIONS) do
    local nodeId=OUTPUT_NODE_OFFSET+actionIndex
    local point=nodePositions[nodeId]
    local selected=action and action.name==option.name
    hudText(guiApi,point.x+4,point.y-3,outputLabels[actionIndex],
      selected and "yellow" or "white","black")
  end

  local buttonNames={"R","L","A","B","U","D"}
  local buttonKeys={"right","left","A","B","up","down"}
  hudText(guiApi,2,113,"PAD","white","black")
  for buttonIndex,buttonName in ipairs(buttonNames) do
    local x=25+(buttonIndex-1)*17
    local pressed=buttons and buttons[buttonKeys[buttonIndex]]
    if pressed then hudBox(guiApi,x-2,111,x+8,123,0xFF196A4C,0xFF65DAA5) end
    hudText(guiApi,x,113,buttonName,pressed and "white" or "gray","black")
  end
  return true
end

local function processHudClick()
  if HUD_CLICK_COOLDOWN>0 then HUD_CLICK_COOLDOWN=HUD_CLICK_COOLDOWN-1;return end
  if not input or not input.get then return end
  local mouse=input.get()
  if mouse and mouse.click==1 and mouse.xmouse>=190 and mouse.xmouse<=255
    and mouse.ymouse>=8 and mouse.ymouse<=26 then
    SHOW_NEURAL_INSPECTOR=not SHOW_NEURAL_INSPECTOR
    HUD_CLICK_COOLDOWN=10
  end
end

function AI.finishEpisode(aiState,state,forced_reason)
  if not aiState.episodeActive then return nil end
  local genome=aiState.populationState.genomes[aiState.genomeIndex]
  local progress=math.max(0,(aiState.furthestWorldX or state.worldX)-(aiState.startWorldX or state.worldX))
  local survival=math.min(aiState.episodeFrames,12000)*0.02
  local power=(aiState.bestForm or 0)*150
  local fitness=progress*10+survival+power+math.max(0,aiState.episodeReward)
  if state and state.phase=="death" then fitness=fitness-120
  elseif state and state.phase=="victory" then fitness=fitness+10000 end
  if forced_reason=="stuck" then fitness=fitness-20 end
  if forced_reason=="timeout" then fitness=fitness-80 end
  aiState.totalEpisodes=aiState.totalEpisodes+1
  aiState.startWorldX=nil
  aiState.episodeActive=false
  aiState.lastFitness=fitness
  if not aiState.championMode then
    genome.fitness=fitness
    aiState.genomeIndex=aiState.genomeIndex+1
    if aiState.genomeIndex>#aiState.populationState.genomes then
      aiState.populationState=AI.nextGeneration(aiState.populationState)
      aiState.genomeIndex=1
    end
  end
  aiState.finished=true
  return fitness
end

function AI.run()
  assert(memory and memory.readbyte and joypad and joypad.set and emu and emu.frameadvance
    and emu.registerexit,
    "Load Mario AI NEAT in FCEUX with an NES SMB1 ROM open")
  math.randomseed(os.time())
  local databasePath=getDatabasePath()
  local loaded=AI.load(databasePath)
  local aiState=AI.new(loaded or AI.newPopulation())
  local logPath=getLogPath()
  local stateAdapter,stateProblem
  if USE_FIXED_TRAINING_STATE then
    stateAdapter,stateProblem=AI.createStateAdapter(savestate,TRAINING_SAVESTATE_SLOT)
  end
  local fixedTraining=stateAdapter~=nil
  local stateSaved=false
  if PLAY_CHAMPION_ONLY and loaded then
    aiState.championMode=true
    aiState.genomeIndex=AI.bestGenomeIndex(aiState.populationState)
  end
  local function savePopulation(context)
    aiState.databaseOK=AI.save(aiState.populationState,databasePath)
    if not aiState.databaseOK then AI.appendLog("database save failed: "..context,logPath) end
    return aiState.databaseOK
  end
  AI.appendLog(string.format("started | database=%s | generation=%d | population=%d | mode=%s | state=%s | test_timer=%s | test_lives=%s",
    loaded and "loaded" or "new",aiState.populationState.generation,#aiState.populationState.genomes,
    aiState.championMode and "champion" or "training",
    fixedTraining and ("slot "..TRAINING_SAVESTATE_SLOT.." via "..stateAdapter.kind) or (stateProblem or "continuous"),
    TESTING_FREEZE_TIMER and "999" or "off",TESTING_INFINITE_LIVES and "refreshed" or "off"),logPath)
  if not loaded then savePopulation("initial population") end
  emu.registerexit(function()
    savePopulation("FCEUX exit")
    AI.appendLog("stopped | episodes="..aiState.totalEpisodes.." | generation="..aiState.populationState.generation,logPath)
  end)
  local function restoreTrainingState(reason)
    if not fixedTraining then return end
    joypad.set(1,{})
    if stateAdapter:load() then
      AI.appendLog("restored training slot "..TRAINING_SAVESTATE_SLOT.." after "..reason,logPath)
    else
      fixedTraining=false
      AI.appendLog("training slot restore failed; switched to continuous training",logPath)
    end
  end
  local lastSaveFrame=0
  while true do
    local state=AI.observe(aiState.frames+1)
    AI.keepLivesForTesting()
    if state.phase=="playing" and not aiState.episodeActive then
      if fixedTraining and not stateSaved then
        if stateAdapter:save() then
          stateSaved=true
          AI.appendLog("saved fixed training start in slot "..TRAINING_SAVESTATE_SLOT,logPath)
        else
          fixedTraining=false
          AI.appendLog("training slot save failed; switched to continuous training",logPath)
        end
      end
      AI.beginEpisode(aiState,state)
      AI.appendLog(string.format("episode start | generation=%d | genome=%d/%d | x=%d | power=%d",
        aiState.populationState.generation,aiState.genomeIndex,#aiState.populationState.genomes,state.worldX,state.power),logPath)
    end
    if state.phase=="playing" then
      AI.freezeTimerForTesting()
      local action=AI.decide(aiState,state)
      local buttons={}
      for _,name in ipairs({"left","right","up","down","A","B","select"}) do
        if action[name] then buttons[name]=true end
      end
      joypad.set(1,buttons)
      processHudClick()
      if gui then AI.drawNeuralInspector(gui,aiState,state,action,buttons) end
      if aiState.episodeFrames>=12000 or aiState.episodeFrames-aiState.lastProgressFrame>600 then
        local reason=aiState.episodeFrames>=12000 and "timeout" or "stuck"
        local fitness=AI.finishEpisode(aiState,state,reason)
        AI.appendLog(string.format("episode end | reason=%s | fitness=%.2f | max_x=%d | frames=%d",
          reason,fitness or 0,aiState.furthestWorldX or state.worldX,aiState.episodeFrames),logPath)
        if not aiState.championMode then savePopulation("episode "..reason) end
        restoreTrainingState(reason)
      end
    elseif (state.phase=="death" or state.phase=="victory") and aiState.episodeActive then
      local fitness=AI.finishEpisode(aiState,state)
      AI.appendLog(string.format("episode end | reason=%s | fitness=%.2f | max_x=%d | frames=%d",
        state.phase,fitness or 0,aiState.furthestWorldX or state.worldX,aiState.episodeFrames),logPath)
      if not aiState.championMode then savePopulation("episode "..state.phase) end
      restoreTrainingState(state.phase)
    else
      local action=AI.decide(aiState,state)
      joypad.set(1,{})
    end
    if aiState.frames-lastSaveFrame>=SAVE_INTERVAL_FRAMES then
      savePopulation("periodic checkpoint");lastSaveFrame=aiState.frames
    end
    emu.frameadvance()
  end
end

if rawget(_G,"MARIO_AI_TEST") then return AI end
AI.run()
