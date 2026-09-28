local bytes={[0x0770]=1,[0x0772]=1,[0x000E]=8,[0x0086]=80,[0x03AD]=80,
  [0x03B8]=176,[0x0057]=0x20}
for row=0,12 do for col=0,15 do
  if row==10 then bytes[0x0500+row*16+col]=0x54 end
end end
local frames,inputs,writes=0,0,0
local exit_callback
memory={
  readbyte=function(address) return bytes[address] or 0 end,
  writebyte=function() writes=writes+1 end,
}
joypad={set=function(player,buttons) assert(player==1);inputs=inputs+1 end}
gui={text=function() end}
local persist_calls=0
savestate={
  persist=function() persist_calls=persist_calls+1 end,
}
emu={frameadvance=function()
  frames=frames+1
  if frames==20 then error("normal test stop") end
end,registerexit=function(callback) exit_callback=callback end}
local ok,err=pcall(dofile,"mario_ai_heaven.lua")
assert(not ok and tostring(err):find("normal test stop",1,true),tostring(err))
assert(frames==20 and inputs==20,"one input and one advance per AI decision")
assert(writes==0,"game RAM must never be changed")
assert(persist_calls==0,"the bot must not call FCEUX savestate.persist")
assert(type(exit_callback)=="function","stopping FCEUX registers a final database save")
exit_callback()
local db=io.open("mario_ai_heaven_neat.db","r")
assert(db~=nil,"final FCEUX exit saves the evolved population")
db:close();os.remove("mario_ai_heaven_neat.db")
os.remove("mario_ai_heaven.log")
print("FCEUX loop smoke test passed")
