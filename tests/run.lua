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

local bot=Bot.new()
local clear=state()
local action=Bot.decide(bot,clear)
test("forward progress is the default",action.right and action.B and not action.A)

-- A visible pit produces a jump action before Mario reaches its edge.
local gap=state()
for col=6,9 do gap.tiles[10*16+col]=0 end
action=Bot.decide(Bot.new(),gap)
test("jump candidate chosen before a reachable gap",action.A==true)

-- An unseen landing cannot be treated as a safe jump destination.
local pit=state()
for col=6,15 do pit.tiles[10*16+col]=0 end
action=Bot.decide(Bot.new(),pit)
test("unknown or unlandable pit does not trigger a blind jump",action.A~=true)

-- Repeated no-progress output is remembered and replaced with a different
-- candidate rather than issuing the same buttons indefinitely.
local stuck=Bot.new()
local previous
for frame=1,43 do
  local sample=state();sample.frame=frame
  action=Bot.decide(stuck,sample)
  if frame==42 then previous=action.name end
end
test("stuck detector activates",stuck.recovery>0)
test("recovery changes its action",action.name~=previous)
test("recovery uses controller actions only",stuck.recovery>0 and action.reason~=nil)

-- Valuable item pursuit can move backwards when the backward route is safe.
local powerup=state()
powerup.items={{kind="powerup",type=0,x=48,y=190}}
action=Bot.decide(Bot.new(),powerup)
test("bot may backtrack for a reachable mushroom",action.left==true)

-- A rearward enemy is a valid fire objective for powered Mario.
local hunter=state()
hunter.power=2
hunter.enemies={{slot=0,id=6,name="goomba",status=0,x=48,y=192,vx=3}}
action=Bot.decide(Bot.new(),hunter)
assert(action.left==true and action.B==true,"backward attack action was "..tostring(action.name))
checks=checks+1

print(string.format("%d behavior checks passed",checks))
