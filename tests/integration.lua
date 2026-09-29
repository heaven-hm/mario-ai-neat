local bytes={[0x0770]=1,[0x0772]=1,[0x000E]=8,[0x0086]=80,[0x03AD]=80,
  [0x03B8]=176,[0x0057]=0x20}
for row=0,12 do for col=0,15 do
  if row==10 then bytes[0x0500+row*16+col]=0x54 end
end end
local frames,inputs,writes,starts=0,0,0,0
local exit_callback
local starterDatabasePath="mario_ai_neat.db"
local starterDatabaseBackupPath=starterDatabasePath..".test-backup"
local starterDatabase=io.open(starterDatabasePath,"r")
if starterDatabase then
  starterDatabase:close()
  os.rename(starterDatabasePath,starterDatabaseBackupPath)
end
memory={
  readbyte=function(address) return bytes[address] or 0 end,
  writebyte=function() writes=writes+1 end,
}
joypad={set=function(player,buttons)
  assert(player==1);inputs=inputs+1
  if buttons.start then starts=starts+1 end
end}
gui={text=function() end}
local persist_calls=0
local slot_saves,slot_loads=0,0
savestate={
  persist=function() persist_calls=persist_calls+1 end,
  object=function(slot) assert(slot==9);return {slot=slot} end,
  save=function(handle) assert(handle.slot==9);slot_saves=slot_saves+1 end,
  load=function(handle)
    assert(handle.slot==9);slot_loads=slot_loads+1;bytes[0x000E]=8
  end,
}
emu={frameadvance=function()
  frames=frames+1
  if frames==1 then bytes[0x000E]=0x0B end
  if frames==20 then error("normal test stop") end
end,registerexit=function(callback) exit_callback=callback end}
local ok,err=pcall(dofile,"mario_ai_neat.lua")
assert(not ok and tostring(err):find("normal test stop",1,true),tostring(err))
assert(frames==20 and inputs==20,"one input and one advance per AI decision")
assert(writes==frames,"timer runs normally while the lives byte remains refreshed")
assert(persist_calls==0,"AI must not call FCEUX savestate.persist")
assert(slot_saves==1 and slot_loads==0,"an immediate death discards the unsafe training slot")
assert(starts==0,"AI never presses Start during training recovery")
assert(type(exit_callback)=="function","stopping FCEUX registers a final database save")
exit_callback()
local db=io.open("mario_ai_neat.db","r")
assert(db~=nil,"final FCEUX exit saves the evolved population")
db:close();os.remove("mario_ai_neat.db")
os.remove("mario_ai_neat.log")
if starterDatabase then os.rename(starterDatabaseBackupPath,starterDatabasePath) end
print("FCEUX loop smoke test passed")
