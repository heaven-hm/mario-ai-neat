-- Mario AI Heaven: autonomous Super Mario Bros. (NES) controller for FCEUX.
-- Load this single file with a compatible SMB1 ROM open in FCEUX.
-- Production behavior uses controller inputs only; it never edits NES RAM.

local Bot = {}

local RAM = {
  player_state=0x000E, enemy_present=0x000F, enemy_id=0x0016,
  enemy_state=0x001E, player_page=0x006D, enemy_page=0x006E,
  player_x=0x0086, enemy_x=0x0087, player_vx=0x0057,
  player_vy=0x009F, enemy_y=0x00CF, player_y=0x03B8,
  player_screen_x=0x03AD, death_music=0x0712,
  flag_event=0x010E, flag_y=0x070F, tiles=0x0500,
  player_size=0x0754, power=0x0756, operation_mode=0x0770,
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

local NONSOLID = {[0x00]=true,[0x08]=true,[0x24]=true,[0x25]=true,[0x26]=true,
  [0x88]=true,[0xC2]=true,[0xC3]=true,[0xC5]=true}
local STOMPED = {[0x02]=true,[0x03]=true,[0x04]=true,[0x20]=true,[0x22]=true,
  [0x23]=true,[0x83]=true,[0x84]=true,[0xC4]=true}

local function s8(value) return value >= 128 and value - 256 or value end
local function clamp(value, low, high) return math.max(low, math.min(high, value)) end

local function read(address) return memory.readbyte(address) end

function Bot.observe(frame)
  local x=read(RAM.player_page)*256+read(RAM.player_x)
  local mode=read(RAM.operation_mode)
  local player_state=read(RAM.player_state)
  local state={frame=frame,x=x,y=read(RAM.player_y)+16,
    screen_x=read(RAM.player_screen_x),vx=s8(read(RAM.player_vx))/16,
    vy=s8(read(RAM.player_vy)),player_state=player_state,
    size=read(RAM.player_size),power=read(RAM.power),mode=mode,
    enemies={},items={},tiles={},phase="playing"}

  if mode==0 then state.phase="title"
  elseif read(RAM.death_music)==1 or player_state==0x0B then state.phase="death"
  elseif mode~=1 then state.phase="transition"
  elseif player_state~=0x08 then state.phase="locked" end
  if read(RAM.flag_event)==0x3E and read(RAM.flag_y)==0xA0 then state.phase="victory" end

  -- The game keeps two 16-column pages of 16x16 metatiles in the tile buffer.
  -- This mapping must match the disassembly/ROM revision named in docs/ram-map.md.
  for address=0,415 do state.tiles[address]=read(RAM.tiles+address) end
  for slot=0,4 do
    if read(RAM.enemy_present+slot)~=0 then
      local id=read(RAM.enemy_id+slot)
      state.enemies[#state.enemies+1]={slot=slot,id=id,name=ENEMY_NAME[id] or "unknown object",
        status=read(RAM.enemy_state+slot),
        x=read(RAM.enemy_page+slot)*256+read(RAM.enemy_x+slot),
        y=read(RAM.enemy_y+slot)+24,vx=s8(read(0x0058+slot))/16}
    end
  end
  if read(0x0014)==1 and read(0x001B)==0x2E then
    local item_x=math.floor(x/256)*256+read(0x008C)
    if item_x<x-128 then item_x=item_x+256 elseif item_x>x+128 then item_x=item_x-256 end
    state.items[1]={kind="powerup",type=read(0x0039),x=item_x,
      y=read(0x03BE)+16,heading=read(0x004B)}
  end
  return state
end


-- The policy learns with NEAT-style neuroevolution. Each genome controls a
-- real SMB1 play segment; episode fitness selects parents for the next
-- generation. No game RAM is written by the bot or restored by the bot.
local RADIUS = 6
local GRID_WIDTH = RADIUS * 2 + 1
local GRID_INPUTS = GRID_WIDTH * GRID_WIDTH
local GLOBAL_INPUTS = 15
local FEATURE_INPUTS = GRID_INPUTS + GLOBAL_INPUTS
local INPUTS = FEATURE_INPUTS + 1 -- final input is the bias node
local ACTIONS = 6
local MAX_NODES = 1000000
local POPULATION = 100
local SPECIES_THRESHOLD = 1.0
local STALE_SPECIES = 15
local SAVE_EVERY_FRAMES = 600

local ACTIONS_MAP = {
  {name="run", right=true, B=true},
  {name="jump_run", right=true, B=true, A=true},
  {name="retreat", left=true, B=true},
  {name="brake"},
  {name="jump_place", A=true},
  {name="walk", right=true},
}
local NON_STOMPABLE = {[0x07]=true,[0x0C]=true,[0x0D]=true,[0x11]=true,[0x12]=true}

local function safe_write(file, value) file:write(value, "\n") end

function Bot.input_count() return INPUTS end
function Bot.output_node(index) return MAX_NODES + index end
function Bot.sensor_index(dx, dy)
  local col = math.floor((dx + RADIUS * 16) / 16)
  local row = math.floor((dy + RADIUS * 16) / 16)
  return row * GRID_WIDTH + col + 1
end

local function grid_solid(state, dx, dy)
  local world_x=state.x+dx
  local world_y=state.y+dy-16
  local col=math.floor((world_x+8)/16)
  local row=math.floor((world_y-32)/16)
  if row<0 or row>=13 then return false end
  local tile=state.tiles[(math.floor(col/16)%2)*208+row*16+col%16]
  return tile~=nil and tile~=0 and not NONSOLID[tile]
end

local function forward_gap(state)
  for dx=16,64,16 do
    if not grid_solid(state,dx,16) then return 1 end
  end
  return 0
end

local function cell_value(state, dx, dy)
  local world_x = state.x + dx
  local world_y = state.y + dy - 16
  local column = math.floor((world_x + 8) / 16)
  local row = math.floor((world_y - 32) / 16)
  if row < 0 or row >= 13 then return 0 end
  local page = math.floor(column / 16) % 2
  local col = column % 16
  local tile = state.tiles[page * 208 + row * 16 + col]
  local occupied = tile ~= nil and tile ~= 0 and not NONSOLID[tile]
  local value = occupied and 1 or 0
  for _, enemy in ipairs(state.enemies) do
    if not STOMPED[enemy.status]
      and math.abs(enemy.x - (state.x + dx)) <= 8
      and math.abs(enemy.y - (state.y + dy)) <= 8 then
      value = -1
      break
    end
  end
  return value
end

local function nearest_enemy(state)
  local chosen, distance
  for _, enemy in ipairs(state.enemies) do
    if not STOMPED[enemy.status] then
      local dx, dy = enemy.x - state.x, enemy.y - state.y
      local d = math.abs(dx) + math.abs(dy) * 1.5
      if d < 240 and (not distance or d < distance) then chosen, distance = enemy, d end
    end
  end
  return chosen, distance
end

function Bot.inputs(state)
  local input = {}
  for dy=-RADIUS*16,RADIUS*16,16 do
    for dx=-RADIUS*16,RADIUS*16,16 do
      input[#input+1] = cell_value(state,dx,dy)
    end
  end
  local enemy = nearest_enemy(state)
  local dx, dy, evx, kind = 0, 0, 0, 0
  if enemy then
    dx = clamp((enemy.x-state.x)/128,-1,1)
    dy = clamp((enemy.y-state.y)/96,-1,1)
    evx = clamp(enemy.vx/4,-1,1)
    kind = (enemy.id or 0)/51
  end
  input[#input+1] = clamp(state.vx/4,-1,1)
  input[#input+1] = clamp(state.vy/8,-1,1)
  input[#input+1] = state.grounded and 1 or -1
  input[#input+1] = state.size == 1 and -1 or 1
  input[#input+1] = state.power == 2 and 1 or (state.power == 1 and 0 or -1)
  input[#input+1] = dx
  input[#input+1] = dy
  input[#input+1] = evx
  input[#input+1] = kind
  local item = state.items and state.items[1]
  if item then
    input[#input+1] = 1
    input[#input+1] = clamp((item.x-state.x)/160,-1,1)
    input[#input+1] = clamp((item.y-state.y)/96,-1,1)
    input[#input+1] = clamp((item.type or 0)/3,-1,1)
  else
    input[#input+1],input[#input+2],input[#input+3],input[#input+4] = -1,0,0,0
  end
  input[#input+1] = forward_gap(state)
  input[#input+1] = enemy~=nil and enemy.x>state.x and enemy.x-state.x<32 and 1 or 0
  return input
end

local function sigmoid(value)
  value = clamp(value,-60,60)
  return 2/(1+math.exp(-4.9*value))-1
end

function Bot.evaluate(genome, input)
  local values, incoming, nodes = {}, {}, {}
  for i=1,FEATURE_INPUTS do values[i] = input[i] or 0 end
  values[INPUTS] = 1
  nodes[INPUTS] = true
  for i=1,ACTIONS do nodes[MAX_NODES+i] = true end
  for _, gene in ipairs(genome.genes) do
    if gene.enabled then
      incoming[gene.out] = incoming[gene.out] or {}
      incoming[gene.out][#incoming[gene.out]+1] = gene
      nodes[gene.into], nodes[gene.out] = true, true
    end
  end
  local order = {}
  for id in pairs(nodes) do if id > INPUTS then order[#order+1] = id end end
  table.sort(order)
  for _, id in ipairs(order) do
    local links = incoming[id]
    if links then
      local sum = 0
      for _, gene in ipairs(links) do sum = sum + (values[gene.into] or 0)*gene.weight end
      values[id] = sigmoid(sum)
    end
  end
  local output = {}
  for i=1,ACTIONS do output[i] = values[MAX_NODES+i] or 0 end
  return output
end

local function fresh_genome()
  return {genes={},fitness=0,adjustedFitness=0,maxneuron=INPUTS,
    mutationRates={connections=0.8,link=1.0,bias=0.4,node=0.12,enable=0.2,disable=0.2,step=0.1}}
end

local function link_key(into, out) return tostring(into)..":"..tostring(out) end
local function innovation(pool, into, out)
  local key = link_key(into,out)
  if not pool.innovations[key] then
    pool.nextInnovation = pool.nextInnovation + 1
    pool.innovations[key] = pool.nextInnovation
  end
  return pool.innovations[key]
end

local function new_gene(into, out, weight, id)
  return {into=into,out=out,weight=weight,enabled=true,innovation=id}
end

local function contains_link(genome, into, out)
  for _, gene in ipairs(genome.genes) do
    if gene.into==into and gene.out==out then return true end
  end
  return false
end

local function random_node(genome, pool, input_only)
  local candidates = {}
  for i=1,INPUTS do candidates[#candidates+1]=i end
  if not input_only then
    for _, gene in ipairs(genome.genes) do
      if gene.into > INPUTS and gene.into < MAX_NODES then candidates[#candidates+1]=gene.into end
      if gene.out > INPUTS and gene.out < MAX_NODES then candidates[#candidates+1]=gene.out end
    end
  end
  return candidates[math.random(#candidates)]
end

local function link_mutate(genome,pool,bias_only)
  local into = bias_only and INPUTS or random_node(genome,pool,false)
  local out = MAX_NODES + math.random(ACTIONS)
  if not bias_only then
    local hidden = {}
    for _, gene in ipairs(genome.genes) do
      if gene.into > INPUTS and gene.into < MAX_NODES then hidden[#hidden+1]=gene.into end
      if gene.out > INPUTS and gene.out < MAX_NODES then hidden[#hidden+1]=gene.out end
    end
    if #hidden>0 and math.random(2)==1 then out=hidden[math.random(#hidden)] end
  end
  if into >= out or contains_link(genome,into,out) then return end
  table.insert(genome.genes,new_gene(into,out,math.random()*4-2,innovation(pool,into,out)))
end

local function node_mutate(genome,pool)
  local enabled = {}
  for _, gene in ipairs(genome.genes) do if gene.enabled then enabled[#enabled+1]=gene end end
  if #enabled==0 then return end
  local old = enabled[math.random(#enabled)]
  old.enabled=false
  genome.maxneuron=math.max(genome.maxneuron,INPUTS)+1
  if genome.maxneuron>=MAX_NODES then return end
  local node=genome.maxneuron
  table.insert(genome.genes,new_gene(old.into,node,1,innovation(pool,old.into,node)))
  table.insert(genome.genes,new_gene(node,old.out,old.weight,innovation(pool,node,old.out)))
end

local function repeat_mutation(rate, callback)
  while rate > 0 do
    if math.random() < math.min(1,rate) then callback() end
    rate = rate - 1
  end
end

function Bot.mutate(genome,pool)
  for name,rate in pairs(genome.mutationRates) do
    if name~="step" then
      genome.mutationRates[name] = rate * (math.random(2)==1 and 0.95 or 1.05263)
    end
  end
  if math.random() < genome.mutationRates.connections then
    for _, gene in ipairs(genome.genes) do
      if math.random()<0.9 then gene.weight=gene.weight+(math.random()*2-1)*genome.mutationRates.step
      else gene.weight=math.random()*4-2 end
    end
  end
  repeat_mutation(genome.mutationRates.link,function() link_mutate(genome,pool,false) end)
  repeat_mutation(genome.mutationRates.bias,function() link_mutate(genome,pool,true) end)
  repeat_mutation(genome.mutationRates.node,function() node_mutate(genome,pool) end)
  repeat_mutation(genome.mutationRates.enable,function()
    local choices={}; for _,gene in ipairs(genome.genes) do if not gene.enabled then choices[#choices+1]=gene end end
    if #choices>0 then choices[math.random(#choices)].enabled=true end
  end)
  repeat_mutation(genome.mutationRates.disable,function()
    local choices={}; for _,gene in ipairs(genome.genes) do if gene.enabled then choices[#choices+1]=gene end end
    if #choices>0 then choices[math.random(#choices)].enabled=false end
  end)
end

local function copy_genome(genome)
  local copy={genes={},fitness=genome.fitness or 0,adjustedFitness=0,
    maxneuron=genome.maxneuron,mutationRates={}}
  for key,value in pairs(genome.mutationRates) do copy.mutationRates[key]=value end
  for _,gene in ipairs(genome.genes) do
    copy.genes[#copy.genes+1]={into=gene.into,out=gene.out,weight=gene.weight,
      enabled=gene.enabled,innovation=gene.innovation}
  end
  return copy
end

function Bot.newGenome(pool)
  local genome=fresh_genome()
  -- Sparse initial networks keep the first generation diverse and inexpensive.
  for action=1,ACTIONS do
    local into=math.random(INPUTS)
    table.insert(genome.genes,new_gene(into,MAX_NODES+action,
      (math.random()*2-1)*0.5,innovation(pool,into,MAX_NODES+action)))
  end
  return genome
end

local function seeded_genome(pool)
  local genome=fresh_genome()
  local enemy_dx=GRID_INPUTS+6
  local item_dx=GRID_INPUTS+11
  local gap=GRID_INPUTS+14
  local contact=GRID_INPUTS+15
  local function connect(into,action,weight)
    table.insert(genome.genes,new_gene(into,MAX_NODES+action,weight,
      innovation(pool,into,MAX_NODES+action)))
  end
  -- Start from sensible SMB1 play: run on clear ground, jump for an enemy or
  -- pit, and turn back toward a visible reward. Evolution can change all links.
  connect(INPUTS,1,0.55)
  connect(INPUTS,2,-0.28)
  connect(enemy_dx,2,3.4)
  connect(gap,2,3.0)
  connect(item_dx,3,-1.2)
  connect(enemy_dx,3,-2.0)
  connect(INPUTS,5,-0.2)
  connect(contact,5,1.5)
  return genome
end

local function genome_distance(a,b)
  local by_id={}
  for _,gene in ipairs(b.genes) do by_id[gene.innovation]=gene end
  local matches,weight_diff,unmatched=0,0,0
  for _,gene in ipairs(a.genes) do
    local other=by_id[gene.innovation]
    if other then matches=matches+1;weight_diff=weight_diff+math.abs(gene.weight-other.weight)
    else unmatched=unmatched+1 end
  end
  unmatched=unmatched + math.max(0,#b.genes-matches)
  local normalizer=math.max(1,#a.genes,#b.genes)
  local weight=matches>0 and weight_diff/matches or 0
  return 2*unmatched/normalizer + 0.4*weight
end

local function assign_species(pool)
  local old=pool.species or {}
  local species={}
  for _,genome in ipairs(pool.genomes) do
    local match
    for _,group in ipairs(species) do
      if genome_distance(genome,group.representative)<SPECIES_THRESHOLD then match=group;break end
    end
    if not match then
      local prior
      for _,group in ipairs(old) do
        if genome_distance(genome,group.representative)<SPECIES_THRESHOLD then prior=group;break end
      end
      match={id=prior and prior.id or (#species+1),genomes={},topFitness=prior and prior.topFitness or 0,
        staleness=prior and prior.staleness or 0,representative=copy_genome(genome)}
      species[#species+1]=match
    end
    match.genomes[#match.genomes+1]=genome
    genome.species=match.id
  end
  pool.species=species
end

function Bot.newPool(population)
  local pool={generation=1,nextInnovation=ACTIONS,innovations={},genomes={},species={},
    bestFitness=0,population=population or POPULATION}
  for i=1,pool.population do
    local genome
    if i==1 then genome=seeded_genome(pool)
    else genome=copy_genome(pool.genomes[1]);Bot.mutate(genome,pool) end
    pool.genomes[#pool.genomes+1]=genome
  end
  assign_species(pool)
  return pool
end

local function crossover(first,second)
  if second.fitness>first.fitness then first,second=second,first end
  local child=fresh_genome()
  local other={};for _,gene in ipairs(second.genes) do other[gene.innovation]=gene end
  for _,gene in ipairs(first.genes) do
    local match=other[gene.innovation]
    local picked=match and math.random(2)==1 and match or gene
    local copy={into=picked.into,out=picked.out,weight=picked.weight,enabled=picked.enabled,innovation=picked.innovation}
    if match and ((not gene.enabled) or (not match.enabled)) and math.random()<0.75 then copy.enabled=false end
    child.genes[#child.genes+1]=copy
  end
  child.maxneuron=math.max(first.maxneuron,second.maxneuron)
  for key,value in pairs(first.mutationRates) do child.mutationRates[key]=value end
  return child
end

local function rank_species(pool)
  table.sort(pool.genomes,function(a,b)return a.fitness>b.fitness end)
  for rank,genome in ipairs(pool.genomes) do genome.globalRank=#pool.genomes-rank+1 end
  local champion=pool.genomes[1]
  pool.bestFitness=math.max(pool.bestFitness or 0,champion and champion.fitness or 0)
  for _,group in ipairs(pool.species) do
    table.sort(group.genomes,function(a,b)return a.fitness>b.fitness end)
    local top=group.genomes[1] and group.genomes[1].fitness or 0
    if top>group.topFitness then group.topFitness=top;group.staleness=0
    else group.staleness=(group.staleness or 0)+1 end
    local sum=0
    for _,genome in ipairs(group.genomes) do
      genome.adjustedFitness=genome.globalRank/math.max(1,#group.genomes)
      sum=sum+genome.adjustedFitness
    end
    group.averageFitness=sum/math.max(1,#group.genomes)
  end
end

local function choose_species(species)
  local total=0
  for _,group in ipairs(species) do total=total+math.max(0,group.averageFitness or 0) end
  if total<=0 then return species[math.random(#species)] end
  local point=math.random()*total
  for _,group in ipairs(species) do
    point=point-math.max(0,group.averageFitness or 0)
    if point<=0 then return group end
  end
  return species[#species]
end

local function breed_child(group,pool)
  local members=group.genomes
  if #members==1 then
    local child=copy_genome(members[1]);child.fitness=0;child.adjustedFitness=0;return child
  end
  local first=members[math.random(#members)]
  local second=members[math.random(#members)]
  local child=math.random()<0.75 and crossover(first,second) or copy_genome(first)
  child.fitness,child.adjustedFitness=0,0
  Bot.mutate(child,pool)
  return child
end

function Bot.nextGeneration(pool)
  rank_species(pool)
  local champion=pool.genomes[1]
  local kept={}
  for _,group in ipairs(pool.species) do
    if group.staleness<STALE_SPECIES or group.genomes[1]==champion then
      local keep=math.max(1,math.ceil(#group.genomes/2))
      for i=#group.genomes,keep+1,-1 do group.genomes[i]=nil end
      kept[#kept+1]=group
    end
  end
  if #kept==0 then kept={{id=1,genomes={champion},topFitness=champion.fitness,staleness=0,representative=copy_genome(champion),averageFitness=1}} end
  local next_pool={generation=pool.generation+1,nextInnovation=pool.nextInnovation,
    innovations=pool.innovations,species=kept,genomes={copy_genome(champion)},
    bestFitness=pool.bestFitness,population=pool.population}
  local target=pool.population or POPULATION
  while #next_pool.genomes<target do
    local group=choose_species(kept)
    next_pool.genomes[#next_pool.genomes+1]=breed_child(group,next_pool)
  end
  assign_species(next_pool)
  return next_pool
end

local function path_for_database()
  local source=debug and debug.getinfo and debug.getinfo(1,"S").source or ""
  if source:sub(1,1)=="@" then
    local script=source:sub(2)
    local folder=script:match("^(.*[/\\])") or ""
    return folder.."mario_ai_heaven_neat.db"
  end
  return "mario_ai_heaven_neat.db"
end

local function path_for_log()
  local source=debug and debug.getinfo and debug.getinfo(1,"S").source or ""
  if source:sub(1,1)=="@" then
    local script=source:sub(2)
    local folder=script:match("^(.*[/\\])") or ""
    return folder.."mario_ai_heaven.log"
  end
  return "mario_ai_heaven.log"
end

function Bot.appendLog(message,path)
  path=path or path_for_log()
  if not io or not io.open then return false end
  local file=io.open(path,"a")
  if not file then return false end
  local stamp=os and os.date and os.date("%Y-%m-%d %H:%M:%S") or "time-unknown"
  file:write("[",stamp,"] ",tostring(message),"\n")
  file:flush()
  file:close()
  return true
end

function Bot.save(pool,path)
  path=path or path_for_database()
  local temporary=path..".tmp"
  local file=io and io.open and io.open(temporary,"w")
  if not file then return false end
  safe_write(file,table.concat({"MARIO_AI_NEAT_V1",pool.generation,pool.nextInnovation,
    pool.bestFitness or 0,pool.population or #pool.genomes,#pool.genomes},","))
  for i,genome in ipairs(pool.genomes) do
    safe_write(file,table.concat({"G",i,genome.fitness or 0,genome.maxneuron or INPUTS,genome.species or 0},","))
    for key,value in pairs(genome.mutationRates) do safe_write(file,table.concat({"R",i,key,value},",")) end
    for _,gene in ipairs(genome.genes) do
      safe_write(file,table.concat({"N",i,gene.into,gene.out,string.format("%.17g",gene.weight),
        gene.enabled and 1 or 0,gene.innovation},","))
    end
  end
  file:close()
  local ok=os and os.rename and os.rename(temporary,path)
  if not ok then
    local input=io.open(temporary,"r");local output=io.open(path,"w")
    if not input or not output then if input then input:close() end;if output then output:close() end;return false end
    output:write(input:read("*a"));input:close();output:close();os.remove(temporary)
  end
  return true
end

function Bot.load(path)
  path=path or path_for_database()
  if not io or not io.open then return nil end
  local file=io.open(path,"r");if not file then return nil end
  local header=file:read("*l") or ""
  local generation,innovation_id,best,population,count=header:match("^MARIO_AI_NEAT_V1,(%d+),([%d%.]+),([%d%.%-]+),(%d+),(%d+)$")
  if not generation then file:close();return nil end
  local pool={generation=tonumber(generation),nextInnovation=tonumber(innovation_id),
    bestFitness=tonumber(best),population=tonumber(population),genomes={},species={},innovations={}}
  for i=1,tonumber(count) do pool.genomes[i]=fresh_genome() end
  for i=1,tonumber(count) do pool.genomes[i].fitness=0 end
  for line in file:lines() do
    local fields={}
    for field in (line..","):gmatch("(.-),") do fields[#fields+1]=field end
    if fields[1]=="G" then
      local index=tonumber(fields[2]);if pool.genomes[index] then
        pool.genomes[index].fitness=tonumber(fields[3]) or 0
        pool.genomes[index].maxneuron=tonumber(fields[4]) or INPUTS
        pool.genomes[index].species=tonumber(fields[5]) or 0
      end
    elseif fields[1]=="R" then
      local index=tonumber(fields[2]);if pool.genomes[index] then
        pool.genomes[index].mutationRates[fields[3]]=tonumber(fields[4]) or 0
      end
    elseif fields[1]=="N" then
      local index=tonumber(fields[2]);if pool.genomes[index] then
        local gene={into=tonumber(fields[3]),out=tonumber(fields[4]),weight=tonumber(fields[5]),
          enabled=tonumber(fields[6])==1,innovation=tonumber(fields[7])}
        pool.genomes[index].genes[#pool.genomes[index].genes+1]=gene
        pool.innovations[link_key(gene.into,gene.out)]=gene.innovation
      end
    end
  end
  file:close()
  assign_species(pool)
  return pool
end

local function closest_threat(state)
  local enemy,distance
  for _,item in ipairs(state.enemies) do
    local dx=item.x-state.x
    local dy=math.abs(item.y-state.y)
    if not STOMPED[item.status] and dy<72 and dx>-32 and dx<144 then
      local d=math.abs(dx)+dy
      if not distance or d<distance then enemy,distance=item,d end
    end
  end
  return enemy
end

local function allowed_actions(state,enemy)
  local allowed={}
  if enemy then
    local dx=enemy.x-state.x
    if dx>0 and dx<112 then
      -- Preserve the original bot's stomp response for ground enemies. Close
      -- contact and non-stompable enemies remove forward motion from the policy.
      allowed[3],allowed[4]=true,true
      if not NON_STOMPABLE[enemy.id] and dx>=28 then
        allowed[2]=true
        if state.grounded then allowed[5]=true end
      elseif state.grounded then
        allowed[5]=true
      end
      if state.power==2 and dx>36 and dx>=28 then allowed[1]=true end
      return allowed
    elseif dx<=0 and dx>-32 then
      allowed[1],allowed[2],allowed[3],allowed[4],allowed[5]=true,true,true,true,true
      return allowed
    end
  end
  for i=1,ACTIONS do allowed[i]=true end
  return allowed
end

local function choose_action(genome,state)
  local enemy=closest_threat(state)
  local allowed=allowed_actions(state,enemy)
  local outputs=Bot.evaluate(genome,Bot.inputs(state))
  if state.power==2 and enemy and enemy.x-state.x>36 then outputs[1]=outputs[1]+0.3 end
  local selected,score
  for i=1,ACTIONS do
    if allowed[i] and (not score or outputs[i]>score) then selected,score=i,outputs[i] end
  end
  return ACTIONS_MAP[selected or 4],selected or 4,enemy,outputs
end

function Bot.new(pool)
  return {pool=pool or Bot.newPool(),genomeIndex=1,episodeFrames=0,
    startX=nil,maxX=nil,lastProgressFrame=0,episodeReward=0,totalEpisodes=0,
    lastAction=nil,frames=0,finished=false,episodeActive=false}
end

function Bot.beginEpisode(bot,state)
  bot.episodeFrames=0;bot.episodeReward=0;bot.startX=state.x;bot.maxX=state.x
  bot.lastProgressFrame=0;bot.finished=false
  bot.episodeActive=true
  bot.bestForm=state.power==2 and 2 or (state.size==1 and 0 or 1)
end

function Bot.decide(bot,state)
  bot.frames=bot.frames+1
  if state.phase~="playing" then
    if (state.phase=="title" or state.phase=="death") and state.frame%90==1 then
      return {start=true,name="start",reason="start or retry"}
    end
    return {reason=state.phase}
  end
  if bot.startX==nil then Bot.beginEpisode(bot,state) end
  bot.episodeFrames=bot.episodeFrames+1
  if state.x>bot.maxX then bot.maxX=state.x;bot.lastProgressFrame=bot.episodeFrames end
  bot.bestForm=math.max(bot.bestForm or 0,state.power==2 and 2 or (state.size==1 and 0 or 1))
  local genome=bot.pool.genomes[bot.genomeIndex]
  local action,index,enemy=choose_action(genome,state)
  action.name=ACTIONS_MAP[index].name
  action.reason=enemy and ("learned "..action.name.." | threat "..enemy.name)
    or ("learned "..action.name)
  bot.lastAction=index
  return action
end

function Bot.finishEpisode(bot,state,forced_reason)
  if not bot.episodeActive then return nil end
  local genome=bot.pool.genomes[bot.genomeIndex]
  local progress=math.max(0,(bot.maxX or state.x)-(bot.startX or state.x))
  local survival=math.min(bot.episodeFrames,12000)*0.02
  local power=(bot.bestForm or 0)*150
  local fitness=progress*10+survival+power+math.max(0,bot.episodeReward)
  if state and state.phase=="death" then fitness=fitness-120
  elseif state and state.phase=="victory" then fitness=fitness+10000 end
  if forced_reason=="stuck" then fitness=fitness-20 end
  if forced_reason=="timeout" then fitness=fitness-80 end
  genome.fitness=fitness
  bot.totalEpisodes=bot.totalEpisodes+1
  bot.genomeIndex=bot.genomeIndex+1
  bot.startX=nil
  bot.episodeActive=false
  if bot.genomeIndex>#bot.pool.genomes then
    bot.pool=Bot.nextGeneration(bot.pool)
    bot.genomeIndex=1
  end
  bot.finished=true
  return fitness
end

function Bot.run()
  assert(memory and memory.readbyte and joypad and joypad.set and emu and emu.frameadvance
    and emu.registerexit,
    "Load Mario AI Heaven in FCEUX with an NES SMB1 ROM open")
  math.randomseed(os.time())
  local db=path_for_database()
  local loaded=Bot.load(db)
  local bot=Bot.new(loaded or Bot.newPool())
  local log_path=path_for_log()
  local function save_pool(context)
    bot.databaseOK=Bot.save(bot.pool,db)
    if not bot.databaseOK then Bot.appendLog("database save failed: "..context,log_path) end
    return bot.databaseOK
  end
  Bot.appendLog(string.format("started | database=%s | generation=%d | population=%d | no savestate API",
    loaded and "loaded" or "new",bot.pool.generation,#bot.pool.genomes),log_path)
  if not loaded then save_pool("initial population") end
  emu.registerexit(function()
    save_pool("FCEUX exit")
    Bot.appendLog("stopped | episodes="..bot.totalEpisodes.." | generation="..bot.pool.generation,log_path)
  end)
  local last_save=0
  while true do
    local state=Bot.observe(bot.frames+1)
    if state.phase=="playing" and not bot.episodeActive then
      Bot.beginEpisode(bot,state)
      Bot.appendLog(string.format("episode start | generation=%d | genome=%d/%d | x=%d | power=%d",
        bot.pool.generation,bot.genomeIndex,#bot.pool.genomes,state.x,state.power),log_path)
    end
    if state.phase=="playing" then
      local action=Bot.decide(bot,state)
      local buttons={}
      for _,name in ipairs({"left","right","up","down","A","B","start","select"}) do
        if action[name] then buttons[name]=true end
      end
      joypad.set(1,buttons)
      if gui and gui.text then
        local gen=bot.pool.generation
        gui.text(8,8,string.format("MARIO AI HEAVEN | NEAT gen %d / genome %d of %d",gen,
          bot.genomeIndex,#bot.pool.genomes),"white","black")
        gui.text(8,18,tostring(action.reason or "learning"),"white","black")
        gui.text(8,28,string.format("best x:%d | episodes:%d",bot.maxX or state.x,bot.totalEpisodes),"white","black")
        gui.text(8,38,"population database: "..(bot.databaseOK==false and "save failed" or "active"),"white","black")
      end
      if bot.episodeFrames>=12000 or bot.episodeFrames-bot.lastProgressFrame>600 then
        local reason=bot.episodeFrames>=12000 and "timeout" or "stuck"
        local fitness=Bot.finishEpisode(bot,state,reason)
        Bot.appendLog(string.format("episode end | reason=%s | fitness=%.2f | max_x=%d | frames=%d",
          reason,fitness or 0,bot.maxX or state.x,bot.episodeFrames),log_path)
        save_pool("episode "..reason)
      end
    elseif (state.phase=="death" or state.phase=="victory") and bot.episodeActive then
      local fitness=Bot.finishEpisode(bot,state)
      Bot.appendLog(string.format("episode end | reason=%s | fitness=%.2f | max_x=%d | frames=%d",
        state.phase,fitness or 0,bot.maxX or state.x,bot.episodeFrames),log_path)
      save_pool("episode "..state.phase)
      joypad.set(1,{})
    else
      local action=Bot.decide(bot,state)
      local buttons={};if action.start then buttons.start=true end
      joypad.set(1,buttons)
    end
    if bot.frames-last_save>=SAVE_EVERY_FRAMES then
      save_pool("periodic checkpoint");last_save=bot.frames
    end
    emu.frameadvance()
  end
end

if rawget(_G,"MARIO_AI_TEST") then return Bot end
Bot.run()
