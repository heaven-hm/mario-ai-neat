MARIO_AI_TEST=true
local AI=dofile("mario_ai_neat.lua")
MARIO_AI_TEST=nil
local checks=0
local function test(name,condition) assert(condition,name);checks=checks+1 end

local function state(x)
  local tiles={}
  for page=0,1 do for row=0,12 do for col=0,15 do
    tiles[page*208+row*16+col]=(row==10 and 0x54 or 0)
  end end end
  return {frame=1,phase="playing",worldX=x or 80,worldY=192,screenX=x or 80,
    horizontalVelocity=2.5,verticalVelocity=0,player_state=8,size=1,power=0,mode=1,
    enemies={},items={},tiles=tiles,grounded=true}
end

math.randomseed(117)
local populationState=AI.newPopulation(12)
test("trainer initializes a population of independently mutated genomes",#populationState.genomes==12)
test("new genomes contain executable action outputs",#populationState.genomes[1].genes>0)

local probe=AI.newGenome(populationState)
probe.genes={{sourceNode=1,targetNode=AI.outputNode(1),weight=2,enabled=true,innovation=9001},
  {sourceNode=AI.inputCount(),targetNode=AI.outputNode(1),weight=0.5,enabled=true,innovation=9002}}
local outputs=AI.evaluateGenome(probe,{[1]=1})
test("genome maps sensor values to controller action scores",outputs[1]>0.5)
local splitPath=AI.newGenome(populationState)
splitPath.genes={
  {sourceNode=1,targetNode=187,weight=1,enabled=true,innovation=9101},
  {sourceNode=187,targetNode=186,weight=1,enabled=true,innovation=9102},
  {sourceNode=186,targetNode=AI.outputNode(1),weight=1,enabled=true,innovation=9103},
}
local splitOutputs=AI.evaluateGenome(splitPath,{[1]=1})
test("a newer hidden node can feed an older hidden target",splitOutputs[1]>0.5)

local input_state=state()
input_state.enemies={{slot=0,id=6,name="goomba",status=0,worldX=input_state.worldX+24,worldY=input_state.worldY,horizontalVelocity=0}}
local inputs=AI.buildObservationInputs(input_state)
test("nearby enemy is encoded as a negative local sensor",inputs[AI.sensorIndex(16,0)]==-1)
test("sensor input has fixed grid and player feature dimensions",#inputs==AI.inputCount()-1)
test("ground support is detected under Mario's feet",AI.isGrounded(state())==true)
local airborne=state();airborne.verticalVelocity=-2
test("airborne Mario is not marked as grounded",AI.isGrounded(airborne)==false)

-- The learned policy may choose among safe responses, but the safety shield
-- removes forward-only actions when an unpowered Mario is about to hit an enemy.
local approaching=state(100)
approaching.enemies={{slot=0,id=6,name="goomba",status=0,worldX=164,worldY=192,horizontalVelocity=0}}
local runner=AI.new(populationState)
local action=AI.decide(runner,approaching)
test("close ground enemy is visible to the controller",action.reason:find("threat goomba",1,true)~=nil)
test("safety shield prevents running into a close enemy",action.name~="run" and action.name~="walk")

local seed=AI.new(AI.newPopulation(1))
action=AI.decide(seed,state())
test("seed policy keeps moving across clear ground",action.name=="run")
local gap=state()
for col=6,9 do gap.tiles[10*16+col]=0 end
action=AI.decide(AI.new(AI.newPopulation(1)),gap)
test("seed policy jumps before an observed gap",action.name=="jump_run")
local nearEnemy=state(100)
nearEnemy.enemies={{slot=0,id=6,name="goomba",status=0,worldX=135,worldY=192,horizontalVelocity=0}}
action=AI.decide(AI.new(AI.newPopulation(1)),nearEnemy)
test("seed policy starts a forward jump before a close ground enemy",action.name=="jump_run")
local close=state(100)
close.enemies={{slot=0,id=6,name="goomba",status=0,worldX=118,worldY=192,horizontalVelocity=0}}
action=AI.decide(AI.new(AI.newPopulation(1)),close)
test("contact-range enemy removes forward and forward-jump actions",
  action.name=="jump_place")
local fireMario=state(100)
fireMario.power=2
fireMario.enemies={{slot=0,id=6,name="goomba",status=0,worldX=180,worldY=192,horizontalVelocity=0}}
action=AI.decide(AI.new(AI.newPopulation(1)),fireMario)
test("Fire Mario keeps firing forward when the enemy is at safe range",action.name=="run" and action.B==true)
local rearThreat=state(100)
rearThreat.power=2
rearThreat.enemies={{slot=0,id=6,name="goomba",status=0,worldX=52,worldY=192,horizontalVelocity=0}}
action=AI.decide(AI.new(AI.newPopulation(1)),rearThreat)
test("Fire Mario can turn back to attack a nearby rear enemy",action.name=="retreat" and action.B==true)

local fitnessState=AI.new(AI.newPopulation(2))
local smallState=state();AI.beginEpisode(fitnessState,smallState)
fitnessState.furthestWorldX=smallState.worldX+10
local death=state();death.phase="death";death.power=0;death.size=1
local fitness=AI.finishEpisode(fitnessState,death)
test("death is penalized and no unearned powerup is added",fitness==-20)
local unsafeEpisode=AI.new(AI.newPopulation(2))
AI.beginEpisode(unsafeEpisode,state(1123))
unsafeEpisode.episodeFrames=6
unsafeEpisode.furthestWorldX=1125
test("six-frame death at the training start is an unsafe checkpoint",
  AI.isUnsafeTrainingStart(unsafeEpisode,death)==true)
AI.abandonEpisode(unsafeEpisode)
test("discarding an unsafe checkpoint preserves the current genome and population",
  unsafeEpisode.genomeIndex==1 and unsafeEpisode.totalEpisodes==0
    and unsafeEpisode.episodeActive==false and unsafeEpisode.startWorldX==nil)
local normalEpisode=AI.new(AI.newPopulation(2))
AI.beginEpisode(normalEpisode,state(40))
normalEpisode.episodeFrames=147
normalEpisode.furthestWorldX=310
test("a normal death remains a scored training episode",
  AI.isUnsafeTrainingStart(normalEpisode,death)==false)
local resumedEpisode=AI.new(AI.newPopulation(3))
AI.beginEpisode(resumedEpisode,state(40))
AI.finishEpisode(resumedEpisode,death)
test("finishing an attempt checkpoints the next unevaluated genome",
  resumedEpisode.genomeIndex==2
    and resumedEpisode.populationState.nextGenomeIndex==2)
local victor=AI.new(AI.newPopulation(2))
AI.beginEpisode(victor,smallState)
local win=state();win.phase="victory"
test("reaching the flag earns a completion bonus",AI.finishEpisode(victor,win)==10000)

local save_path=os.tmpname()
populationState.genomes[1].fitness=100
populationState.nextGenomeIndex=5
test("evolved genome population is saved to a persistent database",AI.save(populationState,save_path)==true)
local restored=AI.load(save_path)
test("generation and genome structure reload from the database",
  restored~=nil and #restored.genomes==12 and #restored.genomes[1].genes>0)
test("training resumes at the next genome after a restart",
  restored.nextGenomeIndex==5 and AI.new(restored).genomeIndex==5)
local nextPopulation=AI.nextGeneration(restored)
test("fitness selection creates a full mutated next generation",
  #nextPopulation.genomes==12 and nextPopulation.generation==restored.generation+1)
test("best-scoring genome survives into the next generation",nextPopulation.genomes[1].fitness==100)
test("a new generation starts evaluating at genome one",nextPopulation.nextGenomeIndex==1)
local savedFile=assert(io.open(save_path,"r"))
local legacyText=savedFile:read("*a");savedFile:close()
legacyText=legacyText:gsub("P,%d+\n","",1)
local legacyFile=assert(io.open(save_path,"w"));legacyFile:write(legacyText);legacyFile:close()
local legacyPopulation=AI.load(save_path)
test("older V1 databases without resume progress still load",
  legacyPopulation~=nil and #legacyPopulation.genomes==12
    and AI.new(legacyPopulation).genomeIndex==1)
os.remove(save_path)

-- A new species may contain only its founder. Its descendants must still
-- explore new weights or links, while the global champion remains intact.
math.randomseed(2006)
local singletonPopulation=AI.newPopulation(12)
local founder=singletonPopulation.genomes[1]
founder.fitness=100
singletonPopulation.species={{id=1,genomes={founder},representative=founder,
  topFitness=0,staleness=0}}
local singletonChildren=AI.nextGeneration(singletonPopulation)
local function geneSignature(genome)
  local parts={}
  for _,gene in ipairs(genome.genes) do
    parts[#parts+1]=table.concat({gene.sourceNode,gene.targetNode,gene.weight,
      tostring(gene.enabled)},":")
  end
  return table.concat(parts,"|")
end
local founderSignature=geneSignature(founder)
local changedChildren=0
for childIndex=2,#singletonChildren.genomes do
  if geneSignature(singletonChildren.genomes[childIndex])~=founderSignature then
    changedChildren=changedChildren+1
  end
end
test("one-member species produces changed descendants",changedChildren>0)
test("accelerated breeding keeps one exact champion and the population size",
  geneSignature(singletonChildren.genomes[1])==founderSignature
    and #singletonChildren.genomes==12)

local slowEpisode=AI.new(AI.newPopulation(1))
AI.beginEpisode(slowEpisode,state(40))
slowEpisode.episodeFrames=179
test("early progress window permits a jump setup",AI.episodeStopReason(slowEpisode,state(40))==nil)
slowEpisode.episodeFrames=180
test("a stationary attempt stops after its initial window",
  AI.episodeStopReason(slowEpisode,state(40))=="stuck")
slowEpisode.furthestWorldX=57
test("a moving attempt continues through the initial window",
  AI.episodeStopReason(slowEpisode,state(57))==nil)

math.randomseed(811)
local sampledPopulation=AI.newPopulation(1)
local globalLinks,gridLinks=0,0
for sampleIndex=1,120 do
  local sampledGenome=AI.newGenome(sampledPopulation)
  sampledGenome.mutationRates={connections=0,link=1,bias=0,node=0,
    enable=0,disable=0,step=0.1}
  local originalGeneCount=#sampledGenome.genes
  AI.mutate(sampledGenome,sampledPopulation)
  if #sampledGenome.genes>originalGeneCount then
    local sourceNode=sampledGenome.genes[#sampledGenome.genes].sourceNode
    if sourceNode>169 then globalLinks=globalLinks+1
    else gridLinks=gridLinks+1 end
  end
end
test("new links favor compact game-state inputs while retaining grid exploration",
  globalLinks>gridLinks/2 and gridLinks>0)

local structuralPopulation=AI.newPopulation(1)
structuralPopulation.nextInnovation=778
local function splitCandidate(sourceNode,innovation)
  local candidate=AI.newGenome(structuralPopulation)
  candidate.genes={{sourceNode=sourceNode,targetNode=AI.outputNode(1),
    weight=0.5,enabled=true,innovation=innovation}}
  candidate.mutationRates={connections=0,link=0,bias=0,node=1.5,
    enable=0,disable=0,step=0.1}
  return candidate
end
math.randomseed(122)
local firstSplit=splitCandidate(1,777)
local matchingSplit=splitCandidate(1,777)
local differentSplit=splitCandidate(2,778)
AI.mutate(firstSplit,structuralPopulation)
local originalHidden=structuralPopulation.splitHistory[777]
AI.mutate(matchingSplit,structuralPopulation)
AI.mutate(differentSplit,structuralPopulation)
test("the same historical split reuses one hidden node across genomes",
  originalHidden~=nil and structuralPopulation.splitHistory[777]==originalHidden
    and matchingSplit.highestHiddenNode>=originalHidden)
test("independent splits receive distinct hidden node numbers",
  structuralPopulation.splitHistory[778]~=nil
    and structuralPopulation.splitHistory[778]~=originalHidden)
local structuralPath=os.tmpname()
test("structural history checkpoint saves",AI.save(structuralPopulation,structuralPath))
local structuralReload=AI.load(structuralPath)
test("structural split history survives checkpoint reload",
  structuralReload~=nil and structuralReload.splitHistory[777]==originalHidden
    and structuralReload.splitHistory[778]==structuralPopulation.splitHistory[778])
os.remove(structuralPath)

local saved_handle,loaded_handle
local modern_state=AI.createStateAdapter({
  object=function(slot) return {slot=slot} end,
  save=function(handle) saved_handle=handle end,
  load=function(handle) loaded_handle=handle end,
},9)
test("modern FCEUX state adapter uses the requested predefined slot",modern_state~=nil and modern_state.kind=="object" and modern_state.handle.slot==9)
test("modern FCEUX state adapter saves and loads without persist",modern_state:save() and modern_state:load() and saved_handle==loaded_handle)
local legacy_state=AI.createStateAdapter({
  create=function(slot) return {slot=slot} end,
  save=function() end,load=function() end,
},9)
test("legacy FCEUX state adapter maps slot 9 to create slot 10",legacy_state~=nil and legacy_state.kind=="create" and legacy_state.handle.slot==10)
populationState.genomes[2].fitness=125
test("champion selection finds the highest-fitness genome",AI.bestGenomeIndex(populationState)==2)

local log_path=os.tmpname()
test("runtime log appends and flushes episode diagnostics",AI.appendLog("test event",log_path)==true)
local log_file=io.open(log_path,"r")
local log_text=log_file:read("*a");log_file:close();os.remove(log_path)
test("runtime log includes its event text",log_text:find("test event",1,true)~=nil)

local timer_writes={}
memory={writebyte=function(address,value) timer_writes[address]=value end}
test("each attempt can start at 999",AI.setTimerTo999()==true)
test("timer digits are initialized to 999",
  timer_writes[0x07F8]==9 and timer_writes[0x07F9]==9 and timer_writes[0x07FA]==9)
test("testing lives counter is refreshed",AI.keepLivesForTesting()==true and timer_writes[0x075A]==9)
memory=nil

local paused=state();paused.phase="death"
test("AI never presses Start after death",AI.decide(AI.new(AI.newPopulation(1)),paused).start==nil)
paused.phase="title"
test("AI never presses Start at the title screen",AI.decide(AI.new(AI.newPopulation(1)),paused).start==nil)

local lesson_state=state(100)
lesson_state.enemies={{slot=0,id=6,name="goomba",status=0,worldX=140,worldY=192,horizontalVelocity=0}}
local lesson=AI.learningStatus(AI.new(AI.newPopulation(1)),lesson_state,{name="jump_run"})
test("learning HUD explains an enemy jump lesson",lesson.lesson=="jump timing to clear an enemy")

local inspectorState=AI.new(AI.newPopulation(1))
local inspectorGameState=state(140)
local inspectorAction=AI.decide(inspectorState,inspectorGameState)
local hudCalls={texts={},boxes=0,lines=0,maxX=0,maxY=0,graphMaxX=0,
  controllerBoxes=0,minControllerX=256}
local fakeHud={
  text=function(_,_,text) hudCalls.texts[#hudCalls.texts+1]=text end,
  drawtext=function(_,_,text) hudCalls.texts[#hudCalls.texts+1]=text end,
  drawbox=function(left,_,right,bottom)
    hudCalls.boxes=hudCalls.boxes+1
    hudCalls.maxX=math.max(hudCalls.maxX,right)
    hudCalls.maxY=math.max(hudCalls.maxY,bottom)
    if bottom<200 then hudCalls.graphMaxX=math.max(hudCalls.graphMaxX,right)
    else
      hudCalls.controllerBoxes=hudCalls.controllerBoxes+1
      hudCalls.minControllerX=math.min(hudCalls.minControllerX,left)
    end
  end,
  drawline=function() hudCalls.lines=hudCalls.lines+1 end,
}
test("live NEAT inspector draws successfully from a game decision",
  AI.drawNeuralInspector(fakeHud,inspectorState,inspectorGameState,inspectorAction,{right=true,B=true})==true)
local hudTextOutput=table.concat(hudCalls.texts," ")
test("live inspector renders one mini controller at the lower right",
  hudCalls.boxes>169 and hudCalls.lines>0 and hudTextOutput:find("PAD",1,true)==nil
    and hudCalls.controllerBoxes>0 and hudCalls.minControllerX==189
    and hudCalls.maxX==255 and hudCalls.maxY==231)
test("small sensor labels stay inside the existing graph panel",
  hudCalls.boxes>500 and hudCalls.graphMaxX==180)
test("network heading says ACTIONS",hudTextOutput:find("ACTIONS",1,true)~=nil
  and hudTextOutput:find("INPUTS",1,true)==nil)

print(string.format("%d NEAT behavior checks passed",checks))
