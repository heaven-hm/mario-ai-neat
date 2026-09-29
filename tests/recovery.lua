-- Run this test from a temporary directory containing mario_ai_neat.lua.
local bytes={[0x0770]=1,[0x000E]=8,[0x006D]=4,[0x0086]=99,
  [0x03AD]=99,[0x03B8]=176,[0x0057]=0x20}
local frames,slotSaves,slotLoads,startPresses=0,0,0,0
for column=0,15 do bytes[0x0500+10*16+column]=0x54 end

memory={readbyte=function(address) return bytes[address] or 0 end,
  writebyte=function(address,value) bytes[address]=value end}
joypad={set=function(_,buttons)
  if buttons.start then startPresses=startPresses+1 end
end}
savestate={object=function(slot) return {slot=slot} end,
  save=function(handle) assert(handle.slot==9);slotSaves=slotSaves+1 end,
  load=function(handle) assert(handle.slot==9);slotLoads=slotLoads+1 end}
emu={registerexit=function() end,frameadvance=function()
  frames=frames+1
  if frames==6 then bytes[0x000E]=0x0B end
  if frames==10 then bytes[0x0770]=2 end
  if frames==12 then
    bytes[0x0770]=1;bytes[0x000E]=8;bytes[0x006D]=0;bytes[0x0086]=40
  end
  if frames==16 then error("recovery test stop") end
end}

local ok,errorMessage=pcall(dofile,"mario_ai_neat.lua")
assert(not ok and tostring(errorMessage):find("recovery test stop",1,true),tostring(errorMessage))
assert(slotSaves==2,"AI must save a new checkpoint after Mario respawns")
assert(slotLoads==0,"AI must never reload a checkpoint that causes immediate death")
assert(startPresses==0,"AI must not press Start during death recovery")
local log=assert(io.open("mario_ai_neat.log","r")):read("*a")
assert(log:find("discarded unsafe training start",1,true),"AI must explain discarded checkpoint")
assert(log:find("Mario respawned",1,true),"AI must log recovery")
print("FCEUX death-loop recovery test passed")
