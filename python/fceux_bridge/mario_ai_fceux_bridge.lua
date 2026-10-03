-- Mario AI FCEUX bridge: exposes SMB1 RAM observations to a Python learner.
-- Python owns replay, neural-network training, checkpoints, and action choice.
-- FCEUX owns only emulation, real controller input, and a fixed training state.

local WORKER_DIRECTORY = "__WORKER_DIRECTORY__"
local ACTION_PROFILE = "__ACTION_PROFILE__"
-- Zero-based SMB1 world.  Python injects one verified title-screen target for
-- every worker before FCEUX starts it.
local TARGET_WORLD_INDEX = __TARGET_WORLD_INDEX__
-- Rainbow learns a duration with each movement action. The DDQN/PPO baselines
-- retain their established six-action, fixed-12-frame protocol.
local ACTION_DURATIONS = ACTION_PROFILE=="rainbow" and {6,12,24} or {12}
local RESPONSE_TIMEOUT_FRAMES = 600
local CURRENT_LEVEL_SLOT = 10

local RAM = {
  game_engine_subroutine=0x000E, enemy_present=0x000F, enemy_id=0x0016,
  enemy_state=0x001E, player_page=0x006D, enemy_page=0x006E,
  player_x=0x0086, enemy_x=0x0087, player_vx=0x0057, player_vy=0x009F,
  enemy_y=0x00CF, player_y=0x03B8, death_music=0x0712,
  tiles=0x0500, player_size=0x0754, power=0x0756,
  operation_mode=0x0770,
  world_select_number=0x076B, world_select_enable=0x07FC,
  -- SMB1 sets this after a completed 8-4.  It turns Goombas into Buzzy
  -- Beetles, so the training bridge must keep it clear when using the title
  -- world selector for independent world-start experiments.
  hard_world_flag=0x076A,
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
  {left=true,B=true,A=true},
}
local ACTION_BASE_COUNT = ACTION_PROFILE=="rainbow" and #ACTIONS or (#ACTIONS-1)
local ACTION_COUNT = ACTION_BASE_COUNT * #ACTION_DURATIONS

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
  if not action or action<0 or action>=ACTION_COUNT then return nil end
  local baseIndex=math.floor(action/#ACTION_DURATIONS)
  local durationIndex=(action%#ACTION_DURATIONS)+1
  local durationFrames=ACTION_DURATIONS[durationIndex]
  local requestedDuration=tonumber(text:match('"duration_frames"%s*:%s*(%d+)'))
  if requestedDuration and requestedDuration~=durationFrames then return nil end
  -- A named table keeps every protocol field bound to its own name. The old
  -- positional unpack silently shifted when fields were appended, which made
  -- the loop's checkpoint flag receive the raw action code.
  return {
    base=baseIndex+1,
    action=action,
    durationFrames=durationFrames,
    reset=text:match('"reset"%s*:%s*true')~=nil,
    hold=text:match('"hold"%s*:%s*true')~=nil,
    advance=text:match('"advance"%s*:%s*true')~=nil,
    campaignReset=text:match('"campaign_reset"%s*:%s*true')~=nil,
    restartWithCheats=text:match('"restart_with_cheats"%s*:%s*(%a+)'),
    checkpoint=text:match('"checkpoint"%s*:%s*true')~=nil,
    restoreFrontier=text:match('"restore_frontier"%s*:%s*true')~=nil,
  }
end

local function phase()
  local mode=read(RAM.operation_mode)
  local subroutine=read(RAM.game_engine_subroutine)
  -- SMB1 GameRoutines dispatches 4=FlagpoleSlide, 5=PlayerEndLevel,
  -- 6=PlayerLoseLife, 8=PlayerCtrlRoutine. $010e and $070f are flagpole
  -- animation/collision coordinates, not a victory event.
  -- https://gist.github.com/1wErt3r/4048722
  if mode==1 and (subroutine==0x04 or subroutine==0x05) then return "victory" end
  if read(RAM.death_music)==1 or subroutine==0x06 or subroutine==0x0B then return "death" end
  if mode~=1 or subroutine~=0x08 then return "waiting" end
  return "playing"
end

if rawget(_G,"MARIO_AI_TEST_PHASE") then return {phase=phase,readCommand=readCommand} end

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
  -- Probe at the ground row, not at Mario's height: when airborne the old
  -- probe sat in empty air and reported a gap on every airborne decision
  -- (measured 497/497 on a clean trace; 89.2%% of airborne replay states here),
  -- which blinded the policy to real pits ahead.
  local groundY=worldY+16
  while groundY<worldY+208 and not solidAt(worldX,groundY) do groundY=groundY+16 end
  for offset=16,96,16 do if not solidAt(worldX+offset,groundY) then gap=1;break end end
  local size,power=read(RAM.player_size),read(RAM.power)
  local worldNumber,levelNumber,areaNumber=read(RAM.world_number),read(RAM.level_number),read(RAM.area_number)
  local globals={clamp(horizontalVelocity/4,-1,1),clamp(verticalVelocity/8,-1,1),grounded and 1 or -1,
    size==1 and -1 or 1,power==2 and 1 or (power==1 and 0 or -1),enemyDX,enemyDY,enemyVelocity,enemyType,
    clamp(worldNumber/7*2-1,-1,1),clamp(levelNumber/3*2-1,-1,1),clamp(areaNumber/31*2-1,-1,1),0,
    gap,nearestEnemy and enemyDX>0 and enemyDX<0.25 and 1 or 0}
  for _,value in ipairs(globals) do features[#features+1]=value end
  return {features=features,worldX=worldX,power=power,world=worldNumber,level=levelNumber,phase=phase(),
    operationMode=read(RAM.operation_mode),playerState=read(RAM.game_engine_subroutine)}
end

local function jsonArray(values)
  local result={}
  for index,value in ipairs(values) do result[index]=string.format("%.8g",value) end
  return "["..table.concat(result,",").."]"
end

local function publish(sequence,snapshot,terminal,reasonOverride)
  local reason=terminal and (reasonOverride or snapshot.phase) or ""
  return writeAtomic("observation.json",string.format(
    '{"sequence":%d,"features":%s,"world_x":%d,"power":%d,"world":%d,"level":%d,"terminal":%s,"reason":"%s"}',
    sequence,jsonArray(snapshot.features),snapshot.worldX,snapshot.power,snapshot.world,snapshot.level,
    terminal and "true" or "false",reason))
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
local ACTION_NAMES={"RUN","JUMP+RUN","BACK","STOP","HOP","WALK","JUMP BACK"}
local hudCache={mode="train",sequence=0,steps=0,replay=0,epsilon=1,action=3,
  updates=0,episodes=0,deaths=0,victories=0,bestX=0,loss=0,
  values={0,0,0,0,0,0},grid={},globals={},hidden={}}
if ACTION_PROFILE=="rainbow" then hudCache.action=10 end

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
  hudCache.mode=text:match('"mode"%s*:%s*"([%w_]+)"') or "train"
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
-- its 256 live encoder activations, with paths to the action-duration values.
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
  hudText(left+77,top+96,string.format("DQN>%d",ACTION_COUNT),0xFFFFFF00)
end

local function drawPythonHud()
  if not gui or not (gui.drawtext or gui.text) then return end
  local hud=readHud()
  drawNetworkInspector(hud)
  local panelLeft,panelTop,panelRight,panelBottom=126,5,255,114
  hudBox(panelLeft,panelTop,panelRight,panelBottom,0xFF102D4A,0xFF4A90E2)
  hudText(panelLeft+4,panelTop+3,
    hud.mode=="eval" and "PYTHON EVAL" or "PYTHON RAINBOW",0xFF00FFFF)
  hudText(panelLeft+4,panelTop+13,string.format("S%d U%d",hud.steps,hud.updates),0xFFFFFFFF)
  hudText(panelLeft+4,panelTop+23,string.format("M%d E%.2f",hud.replay,hud.epsilon),0xFFB8C7E0)
  hudText(panelLeft+4,panelTop+33,string.format("EP%d D%d V%d",hud.episodes,hud.deaths,hud.victories),0xFFFFFFFF)
  hudText(panelLeft+4,panelTop+43,string.format("X%d L%.3f",hud.bestX,hud.loss),0xFFB8C7E0)
  local selectedBase=math.floor(hud.action/#ACTION_DURATIONS)
  local selectedDuration=ACTION_DURATIONS[(hud.action%#ACTION_DURATIONS)+1]
  if ACTION_PROFILE=="rainbow" then
    hudText(panelLeft+4,panelTop+51,string.format("CHOICE %df",selectedDuration),0xFFFFFF00)
  end
  for baseIndex=1,ACTION_BASE_COUNT do
    local name=ACTION_NAMES[baseIndex]
    local rowTop=panelTop+(ACTION_PROFILE=="rainbow" and 60 or 54)
      +(baseIndex-1)*(ACTION_PROFILE=="rainbow" and 7 or 9)
    local selected=selectedBase==baseIndex-1
    local value=-math.huge
    for durationIndex=1,#ACTION_DURATIONS do
      local valueIndex=(baseIndex-1)*#ACTION_DURATIONS+durationIndex
      value=math.max(value,hud.values[valueIndex] or 0)
    end
    if selected then hudBox(panelLeft+3,rowTop,panelRight-3,rowTop+6,0xFF174D38,0xFF42FF70) end
    hudText(panelLeft+6,rowTop,(selected and ">" or " ")..name,selected and 0xFFFFFF00 or 0xFFFFFFFF)
    hudText(panelLeft+(ACTION_PROFILE=="rainbow" and 90 or 73),rowTop,
      string.format("%+.1f",value),selected and 0xFFFFFF00 or 0xFFB8C7E0)
  end
end

local currentLevelHandle=nil
local worldStartHandle=nil
local frontierHandle=nil
if savestate and savestate.object then
  currentLevelHandle=savestate.object(CURRENT_LEVEL_SLOT)
  -- FCEUX exposes predefined slots only through 1..10.  The campaign root
  -- therefore uses an anonymous in-memory state instead of invalid slot 11.
  if savestate.create then
    worldStartHandle=savestate.create()
    frontierHandle=savestate.create()
  end
end
writeAtomic("bridge_started.json", '{"bridge":"started"}')
local initialStateSaved=false
local initialStartAttempts=0
local sequence=0
local waitingFrames=0
local groundedStartFrames=0
local groundedStartX=nil
local initialWorldX=nil
local advancingToLevel=nil
local frontierAvailable=false

while true do
  local snapshot=observe()
  drawPythonHud()
  if not initialStateSaved then
    if snapshot.phase=="playing" and snapshot.worldX<=128 and snapshot.features[172]==1 then
      -- Save only after Mario has settled on the ground at the level start.
      -- A merely controllable frame can still be part of the entry/fall
      -- animation; reusing that as the reset point creates inconsistent runs.
      if groundedStartX==snapshot.worldX then
        groundedStartFrames=groundedStartFrames+1
      else
        groundedStartX=snapshot.worldX
        groundedStartFrames=1
      end
      if groundedStartFrames>=8 and currentLevelHandle then
        -- World select is only a start mechanism.  Clear second-quest mode
        -- before preserving the reusable level-start state.
        memory.writebyte(RAM.hard_world_flag,0)
        joypad.set(1,{})
        savestate.save(currentLevelHandle)
        if snapshot.level==0 and worldStartHandle then savestate.save(worldStartHandle) end
        frontierAvailable=false
        initialStateSaved=true
        initialWorldX=snapshot.worldX
      end
    else
      groundedStartFrames=0
      groundedStartX=nil
    end
    if not initialStateSaved then
      waitingFrames=waitingFrames+1
      if waitingFrames%30==0 then publishWaitingStatus(snapshot) end
      -- Make at most three title-screen Start attempts before the first saved
      -- playable frame.  Once that state exists, this branch is never used
      -- again, including after a death or game-over screen.
      -- A fresh FCEUX process can restore SMB1 in the title/game-over wait
      -- state with operation mode 1 and player state 0.  That state is not
      -- playable, but it is also not the normal title mode (0), so checking
      -- only operationMode used to strand new evaluation windows forever.
      -- Retry Start for both known non-playable startup states; once a clean
      -- grounded frame is captured, this branch is never used again.
      local startupNeedsStart=(snapshot.operationMode==0)
        or (snapshot.operationMode==1 and snapshot.playerState==0)
      if startupNeedsStart and initialStartAttempts<3 and waitingFrames%120==0 then
        -- From the supplied SMB1 disassembly: WorldSelectNumber=$076b,
        -- WorldSelectEnableFlag=$07fc, WorldNumber=$075f, LevelNumber=$075c,
        -- and AreaNumber=$0760.  SMB1's selector starts the selected world at
        -- level 1, so this never pretends to select a later level directly.
        memory.writebyte(RAM.hard_world_flag,0)
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
  elseif advancingToLevel~=nil then
    -- A level win is a campaign transition, not a reset.  Let SMB1 run its
    -- own flagpole/castle sequence, then capture the next level once Mario is
    -- settled at its start.  This avoids guessing later-level RAM areas.
    local isExpectedStart=snapshot.phase=="playing" and snapshot.world==TARGET_WORLD_INDEX
      and snapshot.level==advancingToLevel and snapshot.worldX<=128 and snapshot.features[172]==1
    if isExpectedStart then
      if groundedStartX==snapshot.worldX then
        groundedStartFrames=groundedStartFrames+1
      else
        groundedStartX=snapshot.worldX
        groundedStartFrames=1
      end
      if groundedStartFrames>=8 and currentLevelHandle then
        memory.writebyte(RAM.hard_world_flag,0)
        joypad.set(1,{})
        savestate.save(currentLevelHandle)
        initialWorldX=snapshot.worldX
        advancingToLevel=nil
        groundedStartFrames=0
        frontierAvailable=false
      end
    else
      groundedStartFrames=0
      groundedStartX=nil
    end
    joypad.set(1,{})
    drawPythonHud()
    emu.frameadvance()
  else
    -- Never ask the policy to act on a transition/title/death frame. Some
    -- SMB1 death states skip the explicit death marker between observations;
    -- the player being returned before the saved spawn is a reliable fallback.
    local deathReset=snapshot.phase=="waiting" and initialWorldX~=nil
      and snapshot.worldX<initialWorldX-8
    if snapshot.phase=="waiting" and not deathReset then
      joypad.set(1,{})
      drawPythonHud()
      emu.frameadvance()
    else
      sequence=sequence+1
      local terminal=snapshot.phase=="death" or snapshot.phase=="victory" or deathReset
      publish(sequence,snapshot,terminal,deathReset and "death" or nil)
      -- readCommand returns a named table; extract fields from it
      local action,durationFrames,reset,hold,advance,campaignReset,restartWithCheats,checkpoint,restoreFrontier
      local command
      for _=1,RESPONSE_TIMEOUT_FRAMES do
        command = readCommand(sequence)
        if command then
          action = command.action
          durationFrames = command.durationFrames
          reset = command.reset
          hold = command.hold
          advance = command.advance
          campaignReset = command.campaignReset
          restartWithCheats = command.restartWithCheats
          checkpoint = command.checkpoint
          restoreFrontier = command.restoreFrontier
        end
        if action~=nil then break end
        joypad.set(1,{})
        drawPythonHud()
        emu.frameadvance()
      end
      if campaignReset and restartWithCheats~=nil then
        -- FCEUX loads its cheat file only at process startup. Ask the Python
        -- supervisor to restart only this worker with the other mode; the
        -- new bridge selects this world's level 1 and makes fresh states.
        writeAtomic("mode_request.json",string.format(
          '{"sequence":%d,"cheats_enabled":%s}',sequence,restartWithCheats))
        joypad.set(1,{})
        drawPythonHud()
        emu.frameadvance()
      elseif campaignReset and worldStartHandle then
        joypad.set(1,{})
        savestate.load(worldStartHandle)
        initialWorldX=nil
        frontierAvailable=false
        drawPythonHud()
        emu.frameadvance()
      elseif advance and snapshot.level<3 then
        -- The actor only sends advance after a verified flagpole victory.
        advancingToLevel=snapshot.level+1
        groundedStartFrames=0
        groundedStartX=nil
        joypad.set(1,{})
        drawPythonHud()
        emu.frameadvance()
      elseif reset and currentLevelHandle then
        joypad.set(1,{})
        if restoreFrontier and frontierAvailable and frontierHandle then
          savestate.load(frontierHandle)
        else
          savestate.load(currentLevelHandle)
        end
        drawPythonHud()
        emu.frameadvance()
      elseif hold and currentLevelHandle then
        joypad.set(1,{})
        savestate.load(currentLevelHandle)
        drawPythonHud()
        emu.frameadvance()
      else
        if checkpoint and frontierHandle then
          savestate.save(frontierHandle)
          frontierAvailable=true
        end
        for _=1,(durationFrames or 12) do
          -- FCEUX consumes joypad.set at each frame boundary. Reapply the
          -- action every frame, just as the Lua NEAT loop does, so a 24-frame
          -- run or jump is held for 24 frames rather than tapped once.
          joypad.set(1,ACTIONS[action or 4])
          drawPythonHud()
          emu.frameadvance()
        end
      end
    end
  end
end
