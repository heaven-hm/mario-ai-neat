-- Mario AI FCEUX bridge: exposes SMB1 RAM observations to a Python learner.
-- Python owns replay, neural-network training, checkpoints, and action choice.
-- FCEUX owns only emulation, real controller input, and a fixed training state.

local WORKER_DIRECTORY = "__WORKER_DIRECTORY__"
local ACTION_REPEAT_FRAMES = 4
local RESPONSE_TIMEOUT_FRAMES = 600
local TRAINING_SLOT = 10

local RAM = {
  player_state=0x000E, enemy_present=0x000F, enemy_id=0x0016,
  enemy_state=0x001E, player_page=0x006D, enemy_page=0x006E,
  player_x=0x0086, enemy_x=0x0087, player_vx=0x0057, player_vy=0x009F,
  enemy_y=0x00CF, player_y=0x03B8, death_music=0x0712, flag_event=0x010E,
  flag_y=0x070F, tiles=0x0500, player_size=0x0754, power=0x0756,
  operation_mode=0x0770,
}

local NON_SOLID = {[0x00]=true,[0x08]=true,[0x24]=true,[0x25]=true,[0x26]=true,
  [0x88]=true,[0xC2]=true,[0xC3]=true,[0xC5]=true}
local INACTIVE_ENEMY = {[0x02]=true,[0x03]=true,[0x04]=true,[0x20]=true,[0x22]=true,
  [0x23]=true,[0x83]=true,[0x84]=true,[0xC4]=true}
local ACTIONS = {
  {right=true,B=true}, {right=true,B=true,A=true}, {left=true,B=true}, {}, {A=true}, {right=true},
}

local function clamp(value,low,high) return math.max(low,math.min(high,value)) end
local function signed(value) return value>=128 and value-256 or value end
local function read(address) return memory.readbyte(address) end
local function path(name) return WORKER_DIRECTORY.."/"..name end

local function writeAtomic(name,contents)
  local temporary=path(name..".tmp")
  local handle=io.open(temporary,"w")
  if not handle then return false end
  handle:write(contents);handle:flush();handle:close()
  if os.rename(temporary,path(name)) then return true end
  local destination=io.open(path(name),"w")
  if not destination then return false end
  destination:write(contents);destination:close()
  os.remove(temporary)
  return true
end

local function readCommand(sequence)
  local handle=io.open(path("command.json"),"r")
  if not handle then return nil end
  local text=handle:read("*a");handle:close()
  local commandSequence=tonumber(text:match('"sequence"%s*:%s*(%d+)'))
  if commandSequence~=sequence then return nil end
  local action=tonumber(text:match('"action"%s*:%s*(%d+)'))
  if not action or action<0 or action>=#ACTIONS then return nil end
  return action,text:match('"reset"%s*:%s*true')~=nil
end

local function phase()
  local mode=read(RAM.operation_mode)
  local playerState=read(RAM.player_state)
  if read(RAM.flag_event)==0x3E and read(RAM.flag_y)==0xA0 then return "victory" end
  if read(RAM.death_music)==1 or playerState==0x0B then return "death" end
  if mode~=1 or playerState~=0x08 then return "waiting" end
  return "playing"
end

local function solidAt(worldX,worldY)
  local column=math.floor((worldX+8)/16)
  local row=math.floor((worldY-32)/16)
  if row<0 or row>=13 then return false end
  local index=(math.floor(column/16)%2)*208+row*16+column%16
  local tile=read(RAM.tiles+index)
  return tile~=nil and tile~=0 and not NON_SOLID[tile]
end

local function observe()
  local worldX=read(RAM.player_page)*256+read(RAM.player_x)
  local worldY=read(RAM.player_y)+16
  local horizontalVelocity=signed(read(RAM.player_vx))/16
  local verticalVelocity=signed(read(RAM.player_vy))
  local grounded=verticalVelocity==0 and solidAt(worldX,worldY+16)
  local nearestEnemy=nil
  for slot=0,4 do
    if read(RAM.enemy_present+slot)~=0 and not INACTIVE_ENEMY[read(RAM.enemy_state+slot)] then
      local enemy={id=read(RAM.enemy_id+slot),worldX=read(RAM.enemy_page+slot)*256+read(RAM.enemy_x+slot),
        worldY=read(RAM.enemy_y+slot)+24,velocity=signed(read(0x0058+slot))/16}
      local distance=math.abs(enemy.worldX-worldX)+math.abs(enemy.worldY-worldY)
      if not nearestEnemy or distance<nearestEnemy.distance then enemy.distance=distance;nearestEnemy=enemy end
    end
  end
  local features={}
  for vertical=-96,96,16 do
    for horizontal=-96,96,16 do
      local value=solidAt(worldX+horizontal,worldY+vertical-16) and 1 or 0
      if nearestEnemy and math.abs((nearestEnemy.worldX-worldX)-horizontal)<8
        and math.abs((nearestEnemy.worldY-worldY)-vertical)<8 then value=-1 end
      features[#features+1]=value
    end
  end
  local enemyDX,enemyDY,enemyVelocity,enemyType=0,0,0,0
  if nearestEnemy then
    enemyDX=clamp((nearestEnemy.worldX-worldX)/128,-1,1)
    enemyDY=clamp((nearestEnemy.worldY-worldY)/96,-1,1)
    enemyVelocity=clamp(nearestEnemy.velocity/4,-1,1)
    enemyType=nearestEnemy.id/51
  end
  local gap=0
  for offset=16,96,16 do if not solidAt(worldX+offset,worldY+16) then gap=1;break end end
  local size,power=read(RAM.player_size),read(RAM.power)
  local globals={clamp(horizontalVelocity/4,-1,1),clamp(verticalVelocity/8,-1,1),grounded and 1 or -1,
    size==1 and -1 or 1,power==2 and 1 or (power==1 and 0 or -1),enemyDX,enemyDY,enemyVelocity,enemyType,
    -1,0,0,0,gap,nearestEnemy and enemyDX>0 and enemyDX<0.25 and 1 or 0}
  for _,value in ipairs(globals) do features[#features+1]=value end
  return {features=features,worldX=worldX,power=power,phase=phase()}
end

local function jsonArray(values)
  local result={}
  for index,value in ipairs(values) do result[index]=string.format("%.8g",value) end
  return "["..table.concat(result,",").."]"
end

local function publish(sequence,snapshot,terminal)
  local reason=terminal and snapshot.phase or ""
  return writeAtomic("observation.json",string.format(
    '{"sequence":%d,"features":%s,"world_x":%d,"power":%d,"terminal":%s,"reason":"%s"}',
    sequence,jsonArray(snapshot.features),snapshot.worldX,snapshot.power,terminal and "true" or "false",reason))
end

local stateHandle=nil
if savestate and savestate.object then stateHandle=savestate.object(TRAINING_SLOT) end
local initialStateSaved=false
local sequence=0

while true do
  local snapshot=observe()
  if not initialStateSaved then
    if snapshot.phase=="playing" and snapshot.worldX<=128 and stateHandle then
      savestate.save(stateHandle)
      initialStateSaved=true
    else
      joypad.set(1,{})
      emu.frameadvance()
      -- Start SMB1 manually once. The bridge never presses Start for you.
    end
  else
    sequence=sequence+1
    local terminal=snapshot.phase=="death" or snapshot.phase=="victory"
    publish(sequence,snapshot,terminal)
    local action,reset=nil,false
    for _=1,RESPONSE_TIMEOUT_FRAMES do
      action,reset=readCommand(sequence)
      if action~=nil then break end
      joypad.set(1,{})
      emu.frameadvance()
    end
    if reset and stateHandle then
      joypad.set(1,{})
      savestate.load(stateHandle)
      emu.frameadvance()
    else
      joypad.set(1,ACTIONS[(action or 3)+1])
      for _=1,ACTION_REPEAT_FRAMES do emu.frameadvance() end
    end
  end
end
