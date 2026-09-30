-- A champion must let the flagpole sequence finish and enter the next level.
MARIO_AI_TEST=true
local AI=dofile("mario_ai_neat.lua")
MARIO_AI_TEST=nil
assert(AI.save(AI.newPopulation(1),"mario_ai_neat.db"))
local source=assert(io.open("mario_ai_neat.lua","r")):read("*a")
local championSource,replacements=source:gsub("local PLAY_CHAMPION_ONLY = false",
  "local PLAY_CHAMPION_ONLY = true",1)
assert(replacements==1)
local championFile=assert(io.open("champion.lua","w"))
championFile:write(championSource);championFile:close()

local bytes={[0x0770]=1,[0x000E]=8,[0x006D]=0,[0x0086]=40,
  [0x03AD]=40,[0x03B8]=176,[0x0057]=0}
for row=0,12 do for column=0,31 do
  if row==10 then
    bytes[0x0500+math.floor(column/16)*208+row*16+column%16]=0x54
  end
end end
local frame,timerWrites,stateCalls,startPresses=0,0,0,0
local buttonsByFrame={}
memory={readbyte=function(address) return bytes[address] or 0 end,
  writebyte=function(address,value)
    if address>=0x07F8 and address<=0x07FA then timerWrites=timerWrites+1 end
    bytes[address]=value
  end}
joypad={set=function(_,buttons)
  buttonsByFrame[frame]=buttons
  if buttons.start then startPresses=startPresses+1 end
end}
savestate={object=function() stateCalls=stateCalls+1;error("champion requested a training slot") end}
emu={registerexit=function() end,frameadvance=function()
  frame=frame+1
  if frame==3 then
    bytes[0x006D],bytes[0x0086]=11,184
    bytes[0x000E]=4
  end
  if frame==4 then
    bytes[0x000E]=5
    bytes[0x0770]=2
  end
  if frame==7 then bytes[0x0770],bytes[0x000E],bytes[0x006D],bytes[0x0086]=1,8,0,40 end
  if frame==10 then error("champion flow stop") end
end}

local ok,errorMessage=pcall(dofile,"champion.lua")
assert(not ok and tostring(errorMessage):find("champion flow stop",1,true),tostring(errorMessage))
assert(stateCalls==0,"champion play must not create or restore a training slot")
assert(timerWrites==0,"champion leaves the SMB1 timer unchanged")
assert(startPresses==0,"champion play must never press Start")
for transitionFrame=3,6 do
  assert(buttonsByFrame[transitionFrame] and next(buttonsByFrame[transitionFrame])==nil,
    "champion must release the controller during the flagpole sequence")
end
local log=assert(io.open("mario_ai_neat.log","r")):read("*a")
local episodeStarts=0
for _ in log:gmatch("episode start") do episodeStarts=episodeStarts+1 end
assert(episodeStarts>=2,"champion must continue into the next level")
assert(log:find("reason=victory",1,true),"flagpole contact must be scored")
assert(not log:find("restored training slot",1,true),"champion must not rewind")
os.remove("mario_ai_neat.db")
os.remove("mario_ai_neat.log")
os.remove("champion.lua")
print("FCEUX champion level transition test passed")
