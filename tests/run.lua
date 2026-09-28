MARIO_AI_TEST=true
local Bot=dofile("mario_ai_heaven.lua")
MARIO_AI_TEST=nil
local checks=0
local function test(name,condition) assert(condition,name);checks=checks+1 end

local function state(x)
  local tiles={}
  for page=0,1 do for row=0,12 do for col=0,15 do
    tiles[page*208+row*16+col]=(row==10 and 0x54 or 0)
  end end end
  return {frame=1,phase="playing",x=x or 80,y=192,screen_x=x or 80,
    vx=2.5,vy=0,player_state=8,size=1,power=0,mode=1,
    enemies={},items={},tiles=tiles,grounded=true}
end

math.randomseed(117)
local pool=Bot.newPool(12)
test("trainer initializes a population of independently mutated genomes",#pool.genomes==12)
test("new genomes contain executable action outputs",#pool.genomes[1].genes>0)

local probe=Bot.newGenome(pool)
probe.genes={{into=1,out=Bot.output_node(1),weight=2,enabled=true,innovation=9001},
  {into=Bot.input_count(),out=Bot.output_node(1),weight=0.5,enabled=true,innovation=9002}}
local outputs=Bot.evaluate(probe,{[1]=1})
test("genome maps sensor values to controller action scores",outputs[1]>0.5)

local input_state=state()
input_state.enemies={{slot=0,id=6,name="goomba",status=0,x=input_state.x+24,y=input_state.y,vx=0}}
local inputs=Bot.inputs(input_state)
test("nearby enemy is encoded as a negative local sensor",inputs[Bot.sensor_index(16,0)]==-1)
test("sensor input has fixed grid and player feature dimensions",#inputs==Bot.input_count()-1)

-- The learned policy may choose among safe responses, but the safety shield
-- removes forward-only actions when an unpowered Mario is about to hit an enemy.
local approaching=state(100)
approaching.enemies={{slot=0,id=6,name="goomba",status=0,x=164,y=192,vx=0}}
local runner=Bot.new(pool)
local action=Bot.decide(runner,approaching)
test("close ground enemy is visible to the controller",action.reason:find("threat goomba",1,true)~=nil)
test("safety shield prevents running into a close enemy",action.name~="run" and action.name~="walk")

local seed=Bot.new(Bot.newPool(1))
action=Bot.decide(seed,state())
test("seed policy keeps moving across clear ground",action.name=="run")
local gap=state()
for col=6,9 do gap.tiles[10*16+col]=0 end
action=Bot.decide(Bot.new(Bot.newPool(1)),gap)
test("seed policy jumps before an observed gap",action.name=="jump_run")
local nearEnemy=state(100)
nearEnemy.enemies={{slot=0,id=6,name="goomba",status=0,x=135,y=192,vx=0}}
action=Bot.decide(Bot.new(Bot.newPool(1)),nearEnemy)
test("seed policy starts a forward jump before a close ground enemy",action.name=="jump_run")
local close=state(100)
close.enemies={{slot=0,id=6,name="goomba",status=0,x=118,y=192,vx=0}}
action=Bot.decide(Bot.new(Bot.newPool(1)),close)
test("contact-range enemy removes forward and forward-jump actions",
  action.name=="jump_place")
local fireMario=state(100)
fireMario.power=2
fireMario.enemies={{slot=0,id=6,name="goomba",status=0,x=180,y=192,vx=0}}
action=Bot.decide(Bot.new(Bot.newPool(1)),fireMario)
test("Fire Mario keeps firing forward when the enemy is at safe range",action.name=="run" and action.B==true)
local rearThreat=state(100)
rearThreat.power=2
rearThreat.enemies={{slot=0,id=6,name="goomba",status=0,x=52,y=192,vx=0}}
action=Bot.decide(Bot.new(Bot.newPool(1)),rearThreat)
test("Fire Mario can turn back to attack a nearby rear enemy",action.name=="retreat" and action.B==true)

local fitness_bot=Bot.new(Bot.newPool(2))
local small=state();Bot.beginEpisode(fitness_bot,small)
fitness_bot.maxX=small.x+10
local death=state();death.phase="death";death.power=0;death.size=1
local fitness=Bot.finishEpisode(fitness_bot,death)
test("death is penalized and no unearned powerup is added",fitness==-20)
local victor=Bot.new(Bot.newPool(2))
Bot.beginEpisode(victor,small)
local win=state();win.phase="victory"
test("reaching the flag earns a completion bonus",Bot.finishEpisode(victor,win)==10000)

local save_path=os.tmpname()
pool.genomes[1].fitness=100
test("evolved genome population is saved to a persistent database",Bot.save(pool,save_path)==true)
local restored=Bot.load(save_path)
test("generation and genome structure reload from the database",
  restored~=nil and #restored.genomes==12 and #restored.genomes[1].genes>0)
local next_pool=Bot.nextGeneration(restored)
test("fitness selection creates a full mutated next generation",
  #next_pool.genomes==12 and next_pool.generation==restored.generation+1)
test("best-scoring genome survives into the next generation",next_pool.genomes[1].fitness==100)
os.remove(save_path)

print(string.format("%d NEAT behavior checks passed",checks))
