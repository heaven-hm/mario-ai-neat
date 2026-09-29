-- Mario AI FCEUX bridge: exposes SMB1 RAM observations to a Python learner.
-- Python owns replay, neural-network training, checkpoints, and action choice.
-- FCEUX owns only emulation, real controller input, and a fixed training state.

local WORKER_DIRECTORY = "__WORKER_DIRECTORY__"
-- Zero-based SMB1 world.  Python injects one verified title-screen target for
-- every worker before FCEUX starts it.
local TARGET_WORLD_INDEX = __TARGET_WORLD_INDEX__
-- SMB1 needs a held A press for a full jump.  Four frames cut jumps short;
-- twelve keeps the same action long enough to clear the first enemy and pipe.
local ACTION_REPEAT_FRAMES = 12
local RESPONSE_TIMEOUT_FRAMES = 600
local TRAINING_SLOT = 10
local TEST_TIMER_DIGIT = 0x09
-- SMB1 stores one less than the displayed lives count: 0x62 displays 99.
local TEST_LIVES_RAW = 0x62

local RAM = {
  player_state=0x000E, enemy_present=0x000F, enemy_id=0x0016,
  enemy_state=0x001E, player_page=0x006D, enemy_page=0x006E,
  player_x=0x0086, enemy_x=0x0087, player_vx=0x0057, player_vy=0x009F,
  enemy_y=0x00CF, player_y=0x03B8, death_music=0x0712, flag_event=0x010E,
  flag_y=0x070F, tiles=0x0500, player_size=0x0754, power=0x0756,
  operation_mode=0x0770,
  world_select_number=0x076B, world_select_enable=0x07FC,
  world_number=0x075F, level_number=0x075C, area_number=0x0760,
  offscreen_world_number=0x0766, offscreen_area_number=0x0767,
  timer_hundreds=0x07F8, timer_tens=0x07F9, timer_ones=0x07FA,
  lives=0x075A,
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

local function applyTestingAids()
  -- These test aids match the Lua NEAT training behaviour.  They affect
  -- episode availability only; the AI receives no artificial movement.
  memory.writebyte(RAM.timer_hundreds,TEST_TIMER_DIGIT)
  memory.writebyte(RAM.timer_tens,TEST_TIMER_DIGIT)
  memory.writebyte(RAM.timer_ones,TEST_TIMER_DIGIT)
  memory.writebyte(RAM.lives,TEST_LIVES_RAW)
end

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
  return action,text:match('"reset"%s*:%s*true')~=nil,text:match('"hold"%s*:%s*true')~=nil
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
  local worldNumber,levelNumber,areaNumber=read(RAM.world_number),read(RAM.level_number),read(RAM.area_number)
  local globals={clamp(horizontalVelocity/4,-1,1),clamp(verticalVelocity/8,-1,1),grounded and 1 or -1,
    size==1 and -1 or 1,power==2 and 1 or (power==1 and 0 or -1),enemyDX,enemyDY,enemyVelocity,enemyType,
    clamp(worldNumber/7*2-1,-1,1),clamp(levelNumber/3*2-1,-1,1),clamp(areaNumber/31*2-1,-1,1),0,
    gap,nearestEnemy and enemyDX>0 and enemyDX<0.25 and 1 or 0}
  for _,value in ipairs(globals) do features[#features+1]=value end
  return {features=features,worldX=worldX,power=power,phase=phase(),
    operationMode=read(RAM.operation_mode),playerState=read(RAM.player_state)}
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

-- Written only while waiting for the initial playable frame.  It makes ROM or
-- FCEUX RAM-map mismatches diagnosable without affecting Python training.
local function publishWaitingStatus(snapshot)
  return writeAtomic("status.json",string.format(
    '{"phase":"%s","world_x":%d,"operation_mode":%d,"player_state":%d}',
    snapshot.phase,snapshot.worldX,snapshot.operationMode,snapshot.playerState))
end

-- Python writes a small telemetry message beside command.json.  The HUD stays
-- inside FCEUX so training can be inspected without opening a terminal.
local ACTION_NAMES={"RUN","JUMP","BACK","STOP","HOP","WALK"}
local hudCache={sequence=0,steps=0,replay=0,epsilon=1,action=3,
  updates=0,episodes=0,deaths=0,victories=0,bestX=0,loss=0,
  values={0,0,0,0,0,0},grid={},globals={},hidden={}}

local function parseNumberArray(text,key,target)
  local encoded=text:match('"'..key..'"%s*:%s*%[([^%]]*)%]')
  if not encoded then return end
  local index=1
  for value in encoded:gmatch('[%-%d%.]+') do
    target[index]=tonumber(value) or 0
    index=index+1
  end
end

local function readHud()
  local handle=io.open(path("hud.json"),"r")
  if not handle then return hudCache end
  local text=handle:read("*a");handle:close()
  local sequence=tonumber(text:match('"sequence"%s*:%s*(%d+)'))
  if not sequence then return hudCache end
  hudCache.sequence=sequence
  hudCache.steps=tonumber(text:match('"steps"%s*:%s*(%d+)')) or hudCache.steps
  hudCache.updates=tonumber(text:match('"updates"%s*:%s*(%d+)')) or hudCache.updates
  hudCache.replay=tonumber(text:match('"replay"%s*:%s*(%d+)')) or hudCache.replay
  hudCache.epsilon=tonumber(text:match('"epsilon"%s*:%s*([%d%.%-]+)')) or hudCache.epsilon
  hudCache.episodes=tonumber(text:match('"episodes"%s*:%s*(%d+)')) or hudCache.episodes
  hudCache.deaths=tonumber(text:match('"deaths"%s*:%s*(%d+)')) or hudCache.deaths
  hudCache.victories=tonumber(text:match('"victories"%s*:%s*(%d+)')) or hudCache.victories
  hudCache.bestX=tonumber(text:match('"best_x"%s*:%s*(%d+)')) or hudCache.bestX
  hudCache.loss=tonumber(text:match('"loss"%s*:%s*([%d%.%-]+)')) or hudCache.loss
  hudCache.action=tonumber(text:match('"action"%s*:%s*(%d+)')) or hudCache.action
  parseNumberArray(text,"values",hudCache.values)
  parseNumberArray(text,"grid",hudCache.grid)
  parseNumberArray(text,"globals",hudCache.globals)
  parseNumberArray(text,"hidden",hudCache.hidden)
  return hudCache
end

local function fceuxColor(argb)
  local alpha=math.floor(argb/0x1000000)%256
  local red=math.floor(argb/0x10000)%256
  local green=math.floor(argb/0x100)%256
  local blue=argb%256
  return alpha+blue*0x100+green*0x10000+red*0x1000000
end

local function hudBox(left,top,right,bottom,fill,outline)
  if gui and gui.drawbox then gui.drawbox(left,top,right,bottom,fceuxColor(fill),fceuxColor(outline or fill)) end
end

local function hudText(left,top,value,color)
  if not gui then return end
  if gui.drawtext then gui.drawtext(left,top,tostring(value),fceuxColor(color or 0xFFFFFFFF),0)
  elseif gui.text then gui.text(left,top,tostring(value),"white","black") end
end

local function hudLine(left,top,right,bottom,color)
  if gui and gui.drawline then gui.drawline(left,top,right,bottom,fceuxColor(color)) end
end

local function activationColor(value)
  if value > 0.15 then return 0xFF42FF70 end
  if value < -0.15 then return 0xFFFF6B5E end
  return 0xFF48617A
end

-- The fixed Rainbow network is too dense to draw every 184x512 link.  This
-- shows the actual 13x13 sensor grid, 15 RAM features, and a 4x4 summary of
-- its 256 live encoder activations, with paths to the six Q-value outputs.
local function drawNetworkInspector(hud)
  local left,top,right,bottom=2,5,123,114
  hudBox(left,top,right,bottom,0xFF102D4A,0xFF4A90E2)
  hudText(left+4,top+3,"SENS > ENC",0xFF00FFFF)
  local gridLeft,gridTop,cellSize=left+5,top+17,3
  for row=0,12 do
    for column=0,12 do
      local index=row*13+column+1
      local value=hud.grid[index] or 0
      local fill=value<0 and 0xFFFF6B5E or (value>0 and 0xFF42FF70 or 0xFF263B52)
      hudBox(gridLeft+column*cellSize,gridTop+row*cellSize,
        gridLeft+column*cellSize+1,gridTop+row*cellSize+1,fill,fill)
    end
  end
  -- A few fixed links communicate flow without visual noise from 94k weights.
  for row=0,3 do hudLine(gridLeft+39,gridTop+row*10,68,top+26+row*10,0x8051758C) end
  hudText(left+46,top+17,"RAM",0xFFB8C7E0)
  local labels={{"VX",1},{"VY",2},{"GRD",3},{"PWR",5},{"ENX",6},{"GAP",14}}
  for index,labelSpec in ipairs(labels) do
    local label,value=labelSpec[1],hud.globals[labelSpec[2]] or 0
    local rowTop=top+25+(index-1)*9
    hudText(left+46,rowTop,label,0xFFB8C7E0)
    local width=math.floor(math.min(1,math.abs(value))*11)
    hudBox(left+64,rowTop+2,left+64+width,rowTop+4,activationColor(value),activationColor(value))
  end
  hudText(left+77,top+17,"ENC 256",0xFFB8C7E0)
  for row=0,3 do
    for column=0,3 do
      local value=hud.hidden[row*4+column+1] or 0
      local nodeLeft=left+79+column*8
      local nodeTop=top+28+row*10
      hudBox(nodeLeft,nodeTop,nodeLeft+5,nodeTop+5,activationColor(value),0xFFFFFFFF)
      hudLine(nodeLeft+5,nodeTop+2,right-2,nodeTop+2,0x604A90E2)
    end
  end
  hudText(left+4,top+96,"N184>512>256",0xFFB8C7E0)
  hudText(left+77,top+96,"DQN>6",0xFFFFFF00)
end

local function drawPythonHud()
  if not gui or not (gui.drawtext or gui.text) then return end
  local hud=readHud()
  drawNetworkInspector(hud)
  local panelLeft,panelTop,panelRight,panelBottom=126,5,255,114
  hudBox(panelLeft,panelTop,panelRight,panelBottom,0xFF102D4A,0xFF4A90E2)
  hudText(panelLeft+4,panelTop+3,"PYTHON RAINBOW",0xFF00FFFF)
  hudText(panelLeft+4,panelTop+13,string.format("S%d U%d",hud.steps,hud.updates),0xFFFFFFFF)
  hudText(panelLeft+4,panelTop+23,string.format("M%d E%.2f",hud.replay,hud.epsilon),0xFFB8C7E0)
  hudText(panelLeft+4,panelTop+33,string.format("EP%d D%d V%d",hud.episodes,hud.deaths,hud.victories),0xFFFFFFFF)
  hudText(panelLeft+4,panelTop+43,string.format("X%d L%.3f",hud.bestX,hud.loss),0xFFB8C7E0)
  for index,name in ipairs(ACTION_NAMES) do
    local rowTop=panelTop+54+(index-1)*9
    local selected=hud.action==index-1
    local value=hud.values[index] or 0
    if selected then hudBox(panelLeft+3,rowTop,panelRight-3,rowTop+6,0xFF174D38,0xFF42FF70) end
    hudText(panelLeft+6,rowTop,(selected and ">" or " ")..name,selected and 0xFFFFFF00 or 0xFFFFFFFF)
    hudText(panelLeft+73,rowTop,string.format("%+.2f",value),selected and 0xFFFFFF00 or 0xFFB8C7E0)
  end
end

local stateHandle=nil
if savestate and savestate.object then stateHandle=savestate.object(TRAINING_SLOT) end
writeAtomic("bridge_started.json", '{"bridge":"started"}')
local initialStateSaved=false
local initialStartAttempts=0
local sequence=0
local waitingFrames=0

while true do
  local snapshot=observe()
  drawPythonHud()
  if not initialStateSaved then
    if snapshot.phase=="playing" and snapshot.worldX<=128 and stateHandle then
      savestate.save(stateHandle)
      applyTestingAids()
      initialStateSaved=true
    else
      waitingFrames=waitingFrames+1
      if waitingFrames%30==0 then publishWaitingStatus(snapshot) end
      -- Make at most three title-screen Start attempts before the first saved
      -- playable frame.  Once that state exists, this branch is never used
      -- again, including after a death or game-over screen.
      if snapshot.operationMode==0 and initialStartAttempts<3 and waitingFrames%120==0 then
        -- From the supplied SMB1 disassembly: WorldSelectNumber=$076b,
        -- WorldSelectEnableFlag=$07fc, WorldNumber=$075f, LevelNumber=$075c,
        -- and AreaNumber=$0760.  SMB1's selector starts the selected world at
        -- level 1, so this never pretends to select a later level directly.
        memory.writebyte(RAM.world_select_enable,1)
        memory.writebyte(RAM.world_select_number,TARGET_WORLD_INDEX)
        memory.writebyte(RAM.world_number,TARGET_WORLD_INDEX)
        memory.writebyte(RAM.level_number,0)
        memory.writebyte(RAM.area_number,0)
        memory.writebyte(RAM.offscreen_world_number,TARGET_WORLD_INDEX)
        memory.writebyte(RAM.offscreen_area_number,0)
        joypad.set(1,{start=true})
        initialStartAttempts=initialStartAttempts+1
      else
        joypad.set(1,{})
      end
      drawPythonHud()
      emu.frameadvance()
      -- FCEUX now advances until World 1-1 reaches a controllable state.
    end
  else
    sequence=sequence+1
    local terminal=snapshot.phase=="death" or snapshot.phase=="victory"
    publish(sequence,snapshot,terminal)
    local action,reset,hold=nil,false,false
    for _=1,RESPONSE_TIMEOUT_FRAMES do
      action,reset,hold=readCommand(sequence)
      if action~=nil then break end
      joypad.set(1,{})
      drawPythonHud()
      emu.frameadvance()
    end
    if reset and stateHandle then
      joypad.set(1,{})
      savestate.load(stateHandle)
      applyTestingAids()
      drawPythonHud()
      emu.frameadvance()
    elseif hold and stateHandle then
      joypad.set(1,{})
      savestate.load(stateHandle)
      applyTestingAids()
      drawPythonHud()
      emu.frameadvance()
    else
      joypad.set(1,ACTIONS[(action or 3)+1])
      for _=1,ACTION_REPEAT_FRAMES do
        drawPythonHud()
        emu.frameadvance()
      end
    end
  end
end
