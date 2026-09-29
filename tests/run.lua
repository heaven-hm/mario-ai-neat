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

local input_state=state()
input_state.enemies={{slot=0,id=6,name="goomba",status=0,worldX=input_state.worldX+24,worldY=input_state.worldY,horizontalVelocity=0}}
local inputs=AI.buildObservationInputs(input_state)
test("nearby enemy is encoded as a negative local sensor",inputs[AI.sensorIndex(16,0)]==-1)
test("sensor input has fixed grid and player feature dimensions",#inputs==AI.inputCount()-1)

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
local victor=AI.new(AI.newPopulation(2))
AI.beginEpisode(victor,smallState)
local win=state();win.phase="victory"
test("reaching the flag earns a completion bonus",AI.finishEpisode(victor,win)==10000)

local save_path=os.tmpname()
populationState.genomes[1].fitness=100
test("evolved genome population is saved to a persistent database",AI.save(populationState,save_path)==true)
local restored=AI.load(save_path)
test("generation and genome structure reload from the database",
  restored~=nil and #restored.genomes==12 and #restored.genomes[1].genes>0)
local nextPopulation=AI.nextGeneration(restored)
test("fitness selection creates a full mutated next generation",
  #nextPopulation.genomes==12 and nextPopulation.generation==restored.generation+1)
test("best-scoring genome survives into the next generation",nextPopulation.genomes[1].fitness==100)
os.remove(save_path)

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
test("testing timer writes all SMB1 timer digits",AI.freezeTimerForTesting()==true)
test("testing timer is refreshed to 999",timer_writes[0x07F8]==9 and timer_writes[0x07F9]==9 and timer_writes[0x07FA]==9)
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
local hudCalls={texts={},boxes=0,lines=0,maxX=0,maxY=0}
local fakeHud={
  text=function(_,_,text) hudCalls.texts[#hudCalls.texts+1]=text end,
  drawtext=function(_,_,text) hudCalls.texts[#hudCalls.texts+1]=text end,
  drawbox=function(_,_,right,bottom)
    hudCalls.boxes=hudCalls.boxes+1
    hudCalls.maxX=math.max(hudCalls.maxX,right)
    hudCalls.maxY=math.max(hudCalls.maxY,bottom)
  end,
  drawline=function() hudCalls.lines=hudCalls.lines+1 end,
}
test("live NEAT inspector draws successfully from a game decision",
  AI.drawNeuralInspector(fakeHud,inspectorState,inspectorGameState,inspectorAction,{right=true,B=true})==true)
local hudTextOutput=table.concat(hudCalls.texts," ")
test("live inspector renders the sensor grid and controller in a small corner",
  hudCalls.boxes>=169 and hudCalls.lines>0 and hudTextOutput:find("PAD",1,true)~=nil
    and hudCalls.maxX<=135 and hudCalls.maxY<=125)

print(string.format("%d NEAT behavior checks passed",checks))
