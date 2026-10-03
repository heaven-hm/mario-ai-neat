-- Exercise the real Lua loop with a mid-level launch, safe-start wait, and flagpole transition.
local bytes={[0x0770]=1,[0x000E]=8,[0x006D]=5,[0x0086]=39,
  [0x03AD]=39,[0x03B8]=176,[0x0057]=0}
for row=0,12 do for column=0,31 do
  if row==10 then
    bytes[0x0500+math.floor(column/16)*208+row*16+column%16]=0x54
  end
end end

local frame,slotSaves,slotLoads,timerWrites,startPresses=0,0,0,0,0
local buttonsByFrame={}
memory={
  readbyte=function(address) return bytes[address] or 0 end,
  writebyte=function(address,value)
    if address>=0x07F8 and address<=0x07FA then timerWrites=timerWrites+1 end
    bytes[address]=value
  end,
}
joypad={set=function(_,buttons)
  buttonsByFrame[frame]=buttons
  if buttons.start then startPresses=startPresses+1 end
end}
savestate={
  object=function(slot) assert(slot==9);return {slot=slot} end,
  save=function(handle) assert(handle.slot==9);slotSaves=slotSaves+1;assert(frame==3) end,
  load=function(handle)
    assert(handle.slot==9);slotLoads=slotLoads+1;assert(frame>=9)
    bytes[0x006D],bytes[0x0086]=0,40
  end,
}
emu={registerexit=function() end,frameadvance=function()
  frame=frame+1
  if frame==3 then bytes[0x006D],bytes[0x0086]=0,40 end
  if frame==5 then
    bytes[0x006D],bytes[0x0086]=11,184
    bytes[0x000E]=4
  end
  if frame==6 then
    bytes[0x000E]=5
    bytes[0x0770]=2
  end
  if frame==9 then bytes[0x0770],bytes[0x000E],bytes[0x006D],bytes[0x0086]=1,8,0,40 end
  if frame==12 then error("level flow stop") end
end}

local ok,errorMessage=pcall(dofile,"mario_ai_neat.lua")
assert(not ok and tostring(errorMessage):find("level flow stop",1,true),tostring(errorMessage))
assert(buttonsByFrame[0] and next(buttonsByFrame[0])==nil,
  "mid-level training waits without pressing a controller button")
assert(slotSaves==1,"the fixed start is captured only after Mario reaches level start")
assert(slotLoads==1,"training restores only after the level transition")
assert(timerWrites==0,"training leaves the SMB1 timer unchanged")
assert(startPresses==0,"the AI never presses Start")
local log=assert(io.open("mario_ai_neat.log","r")):read("*a")
assert(log:find("waiting for level start",1,true),"unsafe mid-level launch is reported")
assert(log:find("episode start",1,true),"an episode starts after reaching level start")
assert(log:find("reason=victory",1,true),"flagpole contact is scored")
assert(log:find("flagpole touched; waiting for SMB1 level transition",1,true),
  "the flagpole sequence is allowed to complete")
assert(log:find("restored training slot 9 after victory",1,true),
  "training restarts from the saved beginning after completion")
os.remove("mario_ai_neat.db")
os.remove("mario_ai_neat.log")
print("FCEUX level-start and flagpole flow test passed")
