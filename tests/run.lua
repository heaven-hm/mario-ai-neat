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
test("fixed-start training accepts the level start",
  AI.isValidTrainingStart(state(40))==true)
test("fixed-start training rejects an arbitrary mid-level checkpoint",
  AI.isValidTrainingStart(state(1000))==false)
test("new genomes contain executable action outputs",#populationState.genomes[1].genes>0)

local probe=AI.newGenome(populationState)
probe.genes={{sourceNode=1,targetNode=AI.outputNode(1),weight=2,enabled=true,innovation=9001},
  {sourceNode=AI.inputCount(),targetNode=AI.outputNode(1),weight=0.5,enabled=true,innovation=9002}}
local outputs=AI.evaluateGenome(probe,{[1]=1})
test("genome maps sensor values to controller action scores",outputs[1]>0.5)
local temporalProbe={genes={{sourceNode=AI.memoryInputNode(6),targetNode=AI.outputNode(1),
  weight=2,enabled=true,innovation=9003}}}
local temporalScores=AI.evaluateGenome(temporalProbe,{}, {[6]=1})
test("reserved temporal observations can drive evolved network connections",
  temporalScores[1]>0.5)
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

local pipeAhead=state(100)
for row=7,10 do pipeAhead.tiles[row*16+8]=0x54 end
test("experience context identifies a nearby solid obstacle",
  AI.experienceContextKey(pipeAhead):find("obstacle",1,true)~=nil)
local gapContext=state(100)
for column=6,9 do gapContext.tiles[10*16+column]=0 end
test("experience context distinguishes a gap from a pipe",
  AI.experienceContextKey(gapContext):find("gap",1,true)~=nil)
local enemyContext=AI.experienceContextKey(approaching)
test("experience context includes an approaching enemy",enemyContext:find("enemy",1,true)~=nil)
local experiencePopulation=AI.newPopulation(2)
AI.updateExperienceMemory(experiencePopulation,enemyContext,2,true)
AI.updateExperienceMemory(experiencePopulation,enemyContext,2,true)
local learnedBias=AI.experienceBias(experiencePopulation,enemyContext,1,2)
test("repeated successful maneuver increases its learned action-duration bias",learnedBias>0)
local similarEnemy=state(100)
similarEnemy.enemies={{slot=0,id=6,name="goomba",status=0,worldX=140,worldY=192,horizontalVelocity=0}}
local similarEnemyContext=AI.experienceContextKey(similarEnemy)
test("experience generalizes across nearby enemy distances",
  AI.experienceBias(experiencePopulation,similarEnemyContext,1,2)>0)
local unrelatedGap=state(100)
for column=6,9 do unrelatedGap.tiles[10*16+column]=0 end
test("experience does not leak across unrelated hazard classes",
  AI.experienceBias(experiencePopulation,AI.experienceContextKey(unrelatedGap),1,2)==0)
for _=1,6 do AI.updateExperienceMemory(experiencePopulation,enemyContext,2,false) end
test("repeated failed outcomes reduce the memory preference for that maneuver",
  AI.experienceBias(experiencePopulation,enemyContext,1,2)<learnedBias)
local clearKey=AI.experienceContextKey(state())
local durationMemory=AI.newPopulation(1)
durationMemory.genomes[1].genes={}
for _=1,20 do AI.updateExperienceMemory(durationMemory,clearKey,20,true) end
local durationRunner=AI.new(durationMemory)
local recalledAction=AI.decide(durationRunner,state())
test("recalled evidence can guide both the selected action and jump duration",
  recalledAction.name=="jump_place" and durationRunner.lastHoldFrames==6)
local shieldMemory=AI.newPopulation(1)
shieldMemory.genomes[1].genes={}
local closeContext=AI.experienceContextKey(close)
for _=1,20 do AI.updateExperienceMemory(shieldMemory,closeContext,4,true) end
local shieldRunner=AI.new(shieldMemory)
local shieldedAction=AI.decide(shieldRunner,close)
test("experience bias cannot override the immediate enemy safety filter",
  shieldedAction.name~="run" and shieldedAction.name~="walk" and shieldedAction.name~="jump_run")
local experienceRun=AI.new(experiencePopulation)
AI.beginEpisode(experienceRun,approaching)
AI.decide(experienceRun,approaching)
AI.finishEpisode(experienceRun,{phase="playing",worldX=approaching.worldX},"stuck")
local recordedExperience=false
for _,choices in pairs(experiencePopulation.experienceMemory) do
  for _,evidence in pairs(choices) do
    if evidence.attempts>0 then recordedExperience=true end
  end
end
test("failed hazard encounters are retained as reusable evidence",recordedExperience==true)
local successPopulation=AI.newPopulation(1)
local successRun=AI.new(successPopulation)
AI.beginEpisode(successRun,approaching)
AI.decide(successRun,approaching)
local passedEnemyState=state(180)
passedEnemyState.enemies={{slot=0,id=6,name="goomba",status=0,worldX=164,
  worldY=192,horizontalVelocity=0}}
AI.decide(successRun,passedEnemyState)
local recordedSuccess=false
for _,choices in pairs(successPopulation.experienceMemory) do
  for _,evidence in pairs(choices) do
    if evidence.successes>0 then recordedSuccess=true end
  end
end
test("passing an observed enemy labels the approach actions as reusable successes",
  recordedSuccess==true)

local fitnessState=AI.new(AI.newPopulation(2))
local smallState=state();AI.beginEpisode(fitnessState,smallState)
fitnessState.furthestWorldX=smallState.worldX+10
local death=state();death.phase="death";death.power=0;death.size=1
local fitness=AI.finishEpisode(fitnessState,death)
test("death remains negative after small behavior-exploration bonuses",fitness<0)
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
test("reaching the flag earns the dominant completion bonus",AI.finishEpisode(victor,win)>=10000)
test("completed attempts retain a detailed progress record",
  #victor.populationState.episodeHistory==1
    and victor.populationState.episodeHistory[1].reason=="victory"
    and victor.populationState.episodeHistory[1].startWorldX==smallState.worldX
    and victor.populationState.episodeHistory[1].maxWorldX==smallState.worldX)
test("top-performing checkpoint keeps the complete champion genome",
  #victor.populationState.topPerformers==1
    and #victor.populationState.topPerformers[1].genome.genes==#victor.populationState.genomes[1].genes)

local timedGenome=AI.newGenome(AI.newPopulation(1))
timedGenome.genes={{sourceNode=AI.inputCount(),targetNode=AI.outputNode(8),
  weight=3,enabled=true,innovation=9901}}
local timedState=AI.new({generation=1,population=1,genomes={timedGenome},
  nextGenomeIndex=1,behaviorArchive={}})
local timedAction=AI.decide(timedState,state(80))
test("duration output selects a two-frame action hold",timedState.lastHoldFrames==2)
local heldAction=AI.decide(timedState,state(80))
test("held action is reused between neural evaluations",
  heldAction.name==timedAction.name and heldAction.reason:find("held learned",1,true)~=nil)
local landmarkState=AI.new(AI.newPopulation(1))
AI.decide(landmarkState,state(100))
AI.decide(landmarkState,state(260))
test("crossing new world landmarks adds event-based training feedback",
  landmarkState.episodeReward>=6)
local temporalState=AI.new(AI.newPopulation(1))
local firstTemporalObservation=state(100)
AI.decide(temporalState,firstTemporalObservation)
test("temporal inputs start neutral without a prior observation",
  temporalState.lastTemporalInputs[1]==0)
local nextTemporalObservation=state(100)
nextTemporalObservation.horizontalVelocity=1.25
AI.decide(temporalState,nextTemporalObservation)
test("temporal inputs track changes in Mario's recent movement",
  temporalState.lastTemporalInputs[1]<0)
local onlineLearning=AI.new(AI.newPopulation(1))
onlineLearning.populationState.genomes[1].genes={}
local observedLearning=false
for frameIndex=1,8 do
  AI.decide(onlineLearning,state(80+frameIndex*4))
end
for _,choices in pairs(onlineLearning.populationState.experienceMemory) do
  for _,record in pairs(choices) do
    if (record.qVisits or 0)>0 then observedLearning=true end
  end
end
test("training updates shared action values during a live episode",observedLearning)
local noveltyState=AI.new(AI.newPopulation(1))
AI.beginEpisode(noveltyState,state(80))
noveltyState.episodeFrames=120
local noveltyFitness=AI.finishEpisode(noveltyState,state(80))
test("episode behavior is scored and added to the novelty archive",
  noveltyFitness>0 and #noveltyState.populationState.behaviorArchive==1)

local save_path=os.tmpname()
local qLearningPopulation=AI.newPopulation(1)
local qUpdated,qValue=AI.updateExperienceQ(qLearningPopulation,enemyContext,2,0.8,
  enemyContext,true,0.9)
test("online Q update learns a positive action value from progress feedback",
  qUpdated==true and qValue>0 and AI.experienceBias(qLearningPopulation,enemyContext,1,2)>0)
local similarQBias=AI.experienceBias(qLearningPopulation,similarEnemyContext,1,2)
test("learned action value transfers to a nearby enemy context",similarQBias>0)
test("learned action value does not transfer between different hazard classes",
  AI.experienceBias(qLearningPopulation,AI.experienceContextKey(unrelatedGap),1,2)==0)
populationState.genomes[1].fitness=100
populationState.nextGenomeIndex=5
test("evolved genome population is saved to a persistent database",AI.save(populationState,save_path)==true)
local restored=AI.load(save_path)
test("generation and genome structure reload from the database",
  restored~=nil and #restored.genomes==12 and #restored.genomes[1].genes>0)
test("training resumes at the next genome after a restart",
  restored.nextGenomeIndex==5 and AI.new(restored).genomeIndex==5)
populationState.episodeHistory={{generation=4,genomeIndex=2,fitness=345.625,
  startWorldX=100,maxWorldX=225,frames=87,reason="death",power=1,jumps=2,
  retreats=1,passedEnemies=1,landings=1,powerUps=0,episodeReward=14,novelty=0.5}}
populationState.topPerformers={{generation=4,genomeIndex=2,fitness=345.625,
  maxWorldX=225,progress=125,reason="death",frames=87,power=1,jumps=2,
  retreats=1,passedEnemies=1,landings=1,powerUps=0,episodeReward=14,novelty=0.5,
  genome=AI.newGenome(populationState)}}
populationState.topPerformers[1].genome.fitness=345.625
populationState.scoreOnlyHistoricalBest=987.5
populationState.experienceMemory={[enemyContext]={[2]={attempts=3,successes=2,failures=1,
  rewardMean=1/3,qValue=0.42,qVisits=7}}}
local previousCheckpointFile=assert(io.open(save_path,"r"))
local previousCheckpointText=previousCheckpointFile:read("*a");previousCheckpointFile:close()
test("checkpoint saves the recent episode history and top genome snapshots",
  AI.save(populationState,save_path))
local checkpointTextFile=assert(io.open(save_path,"r"))
local checkpointText=checkpointTextFile:read("*a");checkpointTextFile:close()
local backupFile=assert(io.open(save_path..".bak","r"))
local backupText=backupFile:read("*a");backupFile:close()
test("replacing a checkpoint preserves the previous complete file as a backup",
  checkpointText:find("MARIO_AI_NEAT_V1",1,true)~=nil
    and backupText==previousCheckpointText
    and AI.load(save_path..".bak")~=nil)
local restoredHistory=AI.load(save_path)
test("episode details reload with the population checkpoint",
  #restoredHistory.episodeHistory==1
    and restoredHistory.episodeHistory[1].maxWorldX==225
    and restoredHistory.episodeHistory[1].episodeReward==14)
test("historical top genome reloads with its learned connections",
  #restoredHistory.topPerformers==1
    and restoredHistory.topPerformers[1].fitness==345.625
    and #restoredHistory.topPerformers[1].genome.genes==#populationState.topPerformers[1].genome.genes
    and AI.policySignature(restoredHistory.topPerformers[1].genome)
      ==AI.policySignature(populationState.topPerformers[1].genome)
    and restoredHistory.scoreOnlyHistoricalBest==987.5)
test("successful and failed context/action evidence reloads from the checkpoint",
  restoredHistory.experienceMemory[enemyContext]~=nil
    and restoredHistory.experienceMemory[enemyContext][2].attempts==3
    and restoredHistory.experienceMemory[enemyContext][2].successes==2
    and restoredHistory.experienceMemory[enemyContext][2].failures==1)
test("online Q values survive database save and resume",
  restoredHistory.experienceMemory[enemyContext][2].qValue==0.42
    and restoredHistory.experienceMemory[enemyContext][2].qVisits==7)
populationState.behaviorArchive={{2,1,0,1,0,3}}
test("novelty archive is saved as compact optional database data",AI.save(populationState,save_path))
local restoredArchive=AI.load(save_path)
test("novelty archive resumes with the saved population",
  restoredArchive~=nil and #restoredArchive.behaviorArchive==1
    and restoredArchive.behaviorArchive[1][1]==2
    and restoredArchive.experienceMemory[enemyContext][2].attempts==3)
local nextPopulation=AI.nextGeneration(restored)
test("fitness selection creates a full mutated next generation",
  #nextPopulation.genomes==12 and nextPopulation.generation==restored.generation+1)
local signatures={}
local duplicatePolicy=false
for _,genome in ipairs(nextPopulation.genomes) do
  local signature=AI.policySignature(genome)
  if signatures[signature] then duplicatePolicy=true end
  signatures[signature]=true
end
test("evolved population excludes behaviorally duplicated policies",not duplicatePolicy)
test("best-scoring genome survives into the next generation",nextPopulation.genomes[1].fitness==100)
test("a new generation starts evaluating at genome one",nextPopulation.nextGenomeIndex==1)
local archiveSeed=AI.newPopulation(12)
archiveSeed.genomes[1].fitness=10
local archivedBest=AI.newGenome(archiveSeed)
archivedBest.fitness=900
archiveSeed.topPerformers={{generation=7,genomeIndex=4,fitness=900,
  genome=archivedBest}}
local archiveSeedSignature=AI.policySignature(archivedBest)
local archiveSeedNext=AI.nextGeneration(archiveSeed)
test("the all-time saved top genome seeds the next population",
  AI.policySignature(archiveSeedNext.genomes[1])==archiveSeedSignature
    and archiveSeedNext.genomes[1].fitness==900)
local savedFile=assert(io.open(save_path,"r"))
local legacyText=savedFile:read("*a");savedFile:close()
legacyText=legacyText:gsub("P,%d+\n","",1)
legacyText=legacyText:gsub("X,[^\n]*\n","")
local legacyFile=assert(io.open(save_path,"w"));legacyFile:write(legacyText);legacyFile:close()
local legacyPopulation=AI.load(save_path)
test("older V1 databases without resume progress still load",
  legacyPopulation~=nil and #legacyPopulation.genomes==12
    and AI.new(legacyPopulation).genomeIndex==1
    and AI.experienceMemorySize(legacyPopulation)==0)
local oldExperienceFile=assert(io.open(save_path,"a"))
oldExperienceFile:write("X,",enemyContext,",2,2,1,1,0.0\n");oldExperienceFile:close()
local oldExperiencePopulation=AI.load(save_path)
test("older experience rows load with neutral Q values",
  oldExperiencePopulation.experienceMemory[enemyContext][2].qValue==0
    and oldExperiencePopulation.experienceMemory[enemyContext][2].qVisits==0)
os.remove(save_path);os.remove(save_path..".bak")

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
populationState.topPerformers={{fitness=999,genome=AI.newGenome(populationState),generation=8,genomeIndex=4}}
test("champion selection considers the highest archived genome",
  AI.bestPerformer(populationState).source=="archive"
    and AI.bestPerformer(populationState).fitness==999)

local log_path=os.tmpname()
test("runtime log appends and flushes episode diagnostics",AI.appendLog("test event",log_path)==true)
local log_file=io.open(log_path,"r")
local log_text=log_file:read("*a");log_file:close();os.remove(log_path)
test("runtime log includes its event text",log_text:find("test event",1,true)~=nil)

local legacy_log_path=os.tmpname()
local legacy_log=assert(io.open(legacy_log_path,"w"))
legacy_log:write("[2026-09-29 12:00:00] episode start | generation=4 | genome=1/2 | x=40 | power=0\n",
  "[2026-09-29 12:00:03] episode end | reason=death | fitness=75.50 | max_x=89 | frames=46\n")
legacy_log:close()
local legacyCheckpoint=AI.newPopulation(2)
legacyCheckpoint.generation=4;legacyCheckpoint.nextGenomeIndex=2
legacyCheckpoint.bestFitness=1200
local importedCount,importedGenomes=AI.importLegacyLogHistory(legacyCheckpoint,legacy_log_path)
test("legacy log migration imports exact progress and a matching saved genome",
  importedCount==1 and importedGenomes==1 and #legacyCheckpoint.episodeHistory==1
    and legacyCheckpoint.episodeHistory[1].startWorldX==40
    and legacyCheckpoint.episodeHistory[1].maxWorldX==89
    and legacyCheckpoint.topPerformers[1].fitness==75.5
    and legacyCheckpoint.bestFitness==75.5
    and legacyCheckpoint.scoreOnlyHistoricalBest==1200)
test("legacy log migration marks unavailable historical behavior fields as unknown",
  legacyCheckpoint.episodeHistory[1].jumps==-1
    and legacyCheckpoint.episodeHistory[1].episodeReward==-1)
local repeatedImport=AI.importLegacyLogHistory(legacyCheckpoint,legacy_log_path)
test("legacy log migration cannot duplicate imported episode rows",repeatedImport==0
  and #legacyCheckpoint.episodeHistory==1)
os.remove(legacy_log_path)

local legacy_log_path=os.tmpname()
local legacy_log=assert(io.open(legacy_log_path,"w"))
legacy_log:write("[2026-09-29 12:00:00] episode start | generation=4 | genome=1/2 | x=40 | power=0\n",
  "[2026-09-29 12:00:03] episode end | reason=death | fitness=75.50 | max_x=89 | frames=46\n")
legacy_log:close()
local legacyCheckpoint=AI.newPopulation(2)
legacyCheckpoint.generation=4;legacyCheckpoint.nextGenomeIndex=2
local importedCount,importedGenomes=AI.importLegacyLogHistory(legacyCheckpoint,legacy_log_path)
test("legacy log migration imports exact progress and a matching saved genome",
  importedCount==1 and importedGenomes==1 and #legacyCheckpoint.episodeHistory==1
    and legacyCheckpoint.episodeHistory[1].startWorldX==40
    and legacyCheckpoint.episodeHistory[1].maxWorldX==89
    and legacyCheckpoint.topPerformers[1].fitness==75.5)
test("legacy log migration marks unavailable historical behavior fields as unknown",
  legacyCheckpoint.episodeHistory[1].jumps==-1
    and legacyCheckpoint.episodeHistory[1].episodeReward==-1)
local repeatedImport=AI.importLegacyLogHistory(legacyCheckpoint,legacy_log_path)
test("legacy log migration cannot duplicate imported episode rows",repeatedImport==0
  and #legacyCheckpoint.episodeHistory==1)
os.remove(legacy_log_path)

local timer_writes={}
memory={writebyte=function(address,value) timer_writes[address]=value end}
test("timer aid is disabled",AI.setTimerTo999()==false)
test("lives aid is disabled",AI.keepLivesForTesting()==false)
test("timer and lives RAM are unchanged",next(timer_writes)==nil)
memory=nil

local tileRangeReads,tileSingleReads=0,0
memory={
  readbyte=function(address)
    if address>=0x0500 and address<=0x069F then tileSingleReads=tileSingleReads+1 end
    return address==0x000E and 0x08 or 0
  end,
  readbyterange=function(address,length)
    tileRangeReads=tileRangeReads+1
    assert(address==0x0500 and length==416)
    return string.char(0x54)..string.rep("\0",414)..string.char(0xAB)
  end,
}
local batchedObservation=AI.observe(1)
test("tile observation uses one RAM block read with unchanged zero-based indices",
  tileRangeReads==1 and tileSingleReads==0
    and batchedObservation.tiles[0]==0x54 and batchedObservation.tiles[415]==0xAB)
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
local sensorImage
local imageHud=setmetatable({drawimage=function(imageX,imageY,imageData)
  sensorImage={x=imageX,y=imageY,data=imageData}
end},{__index=fakeHud})
AI.drawNeuralInspector(imageHud,inspectorState,inspectorGameState,inspectorAction,{right=true,B=true})
test("sensor labels use one prebuilt image when FCEUX supports it",
  sensorImage and sensorImage.x==39 and sensorImage.y==46
    and sensorImage.data:sub(1,2)==string.char(255,254)
    and #sensorImage.data==11+77*57*4)
test("network heading says ACTIONS",hudTextOutput:find("ACTIONS",1,true)~=nil
  and hudTextOutput:find("INPUTS",1,true)==nil)

-- Area transition misclassified as death: World 1-2 pipe entry test
MARIO_AI_TEST_PHASE=true
local bridgePath="python/fceux_bridge/mario_ai_fceux_bridge.lua"
local bridgeFile=io.open(bridgePath,"r")
if bridgeFile then bridgeFile:close() else bridgePath="../"..bridgePath end
local bridge=dofile(bridgePath)
MARIO_AI_TEST_PHASE=nil

-- Verify existing death detection: death_music==1 and subroutine==0x06 still trigger death
local deathTestBytes={[0x0770]=1,[0x000E]=8,[0x0712]=0}
memory={readbyte=function(addr) return deathTestBytes[addr] or 0 end}
assert(bridge.phase()=="playing")
deathTestBytes[0x0712]=1
assert(bridge.phase()=="death","death_music==1 must trigger death")
deathTestBytes[0x0712]=0
deathTestBytes[0x000E]=6
assert(bridge.phase()=="death","subroutine==0x06 must trigger death")
deathTestBytes[0x000E]=0x0B
assert(bridge.phase()=="death","subroutine==0x0B must trigger death")
memory=nil

-- Simulate sequence:
-- Frame 1: phase=playing, worldX=3000, subroutine=0x08
-- Frame 2: phase=waiting, worldX=2950, subroutine=0x02 (SideExitPipeEntry)
-- Frame 3: phase=waiting, worldX=2900, subroutine=0x02 (still in pipe)
-- Frame 4: phase=waiting, worldX=24, subroutine=0x07 (PlayerEntrance in Area 2)
local sequenceFrames={
  {phase="playing", worldX=3000, subroutine=0x08},
  {phase="waiting", worldX=2950, subroutine=0x02},
  {phase="waiting", worldX=2900, subroutine=0x02},
  {phase="waiting", worldX=24,   subroutine=0x07},
}

local initialWorldX=3000
local terminalReported=false
local frame4DeathReset,frame4Phase

for i,frame in ipairs(sequenceFrames) do
  local snapshot={
    phase=frame.phase,
    worldX=frame.worldX,
    subroutine=frame.subroutine,
    death_music=0,
  }
  local deathReset=bridge.checkDeathReset(snapshot,initialWorldX)
  local terminal=snapshot.phase=="death" or snapshot.phase=="victory" or deathReset
  local reported=false
  if snapshot.phase=="waiting" and not deathReset then
    reported=false
  else
    reported=terminal
  end
  if reported then terminalReported=true end
  if i==4 then
    frame4DeathReset=deathReset
    frame4Phase=snapshot.phase
  end
end

assert(frame4DeathReset==false,"deathReset must be false")
assert(frame4Phase~="victory","phase must not be victory")
assert(frame4Phase=="waiting","phase must be waiting")
assert(not terminalReported,"no terminal=true reported during this sequence")

test("World 1-2 pipe entry does not trigger false death",
  frame4DeathReset==false and frame4Phase~="victory" and frame4Phase=="waiting"
    and not terminalReported)

print(string.format("%d NEAT behavior checks passed",checks))
