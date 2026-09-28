-- Mario AI Heaven: autonomous Super Mario Bros. (NES) controller for FCEUX.
-- Load this single file with a compatible SMB1 ROM open in FCEUX.
-- Production behavior uses controller inputs only; it never edits NES RAM.

local Bot = {}

local RAM = {
  player_state=0x000E, enemy_present=0x000F, enemy_id=0x0016,
  enemy_state=0x001E, player_page=0x006D, enemy_page=0x006E,
  player_x=0x0086, enemy_x=0x0087, player_vx=0x0057,
  player_vy=0x009F, enemy_y=0x00CF, player_y=0x03B8,
  player_screen_x=0x03AD, death_music=0x0712,
  flag_event=0x010E, flag_y=0x070F, tiles=0x0500,
  player_size=0x0754, power=0x0756, operation_mode=0x0770,
}

local ENEMY_NAME = {
  [0x00]="green koopa", [0x02]="buzzy beetle", [0x03]="red koopa",
  [0x05]="hammer bro", [0x06]="goomba", [0x07]="bloober",
  [0x08]="bullet bill", [0x09]="paratroopa", [0x0A]="cheep-cheep",
  [0x0B]="cheep-cheep", [0x0C]="podoboo", [0x0D]="piranha plant",
  [0x0E]="jumping paratroopa", [0x0F]="red paratroopa",
  [0x10]="flying paratroopa", [0x11]="lakitu", [0x12]="spiny",
  [0x14]="flying cheep-cheep", [0x15]="bowser flame", [0x2D]="bowser",
  [0x33]="bullet bill",
}

local NONSOLID = {[0x00]=true,[0x08]=true,[0x24]=true,[0x25]=true,[0x26]=true,
  [0x88]=true,[0xC2]=true,[0xC3]=true,[0xC5]=true}
local STOMPED = {[0x02]=true,[0x03]=true,[0x04]=true,[0x20]=true,[0x22]=true,
  [0x23]=true,[0x83]=true,[0x84]=true,[0xC4]=true}

local function s8(value) return value >= 128 and value - 256 or value end
local function clamp(value, low, high) return math.max(low, math.min(high, value)) end

local function read(address) return memory.readbyte(address) end

local function tile_at(state, world_x, screen_y)
  local row = math.floor((screen_y - 32) / 16)
  if row < 0 or row >= 13 then return nil end
  local screen_left = state.x - state.screen_x
  if world_x < screen_left - 16 or world_x >= screen_left + 272 then return nil end
  local column = math.floor(world_x / 16)
  local page = math.floor(column / 16) % 2
  local col = column % 16
  return state.tiles[page * 208 + row * 16 + col]
end

local function solid_at(state, x, y)
  local tile = tile_at(state,x,y)
  if tile == nil then return nil end
  return not NONSOLID[tile]
end

function Bot.observe(frame)
  local x=read(RAM.player_page)*256+read(RAM.player_x)
  local mode=read(RAM.operation_mode)
  local player_state=read(RAM.player_state)
  local state={frame=frame,x=x,y=read(RAM.player_y)+16,
    screen_x=read(RAM.player_screen_x),vx=s8(read(RAM.player_vx))/16,
    vy=s8(read(RAM.player_vy)),player_state=player_state,
    size=read(RAM.player_size),power=read(RAM.power),mode=mode,
    enemies={},items={},tiles={},phase="playing"}

  if mode==0 then state.phase="title"
  elseif read(RAM.death_music)==1 or player_state==0x0B then state.phase="death"
  elseif mode~=1 then state.phase="transition"
  elseif player_state~=0x08 then state.phase="locked" end
  if read(RAM.flag_event)==0x3E and read(RAM.flag_y)==0xA0 then state.phase="victory" end

  -- The game keeps two 16-column pages of 16x16 metatiles in the tile buffer.
  -- This mapping must match the disassembly/ROM revision named in docs/ram-map.md.
  for address=0,415 do state.tiles[address]=read(RAM.tiles+address) end
  for slot=0,4 do
    if read(RAM.enemy_present+slot)~=0 then
      local id=read(RAM.enemy_id+slot)
      state.enemies[#state.enemies+1]={slot=slot,id=id,name=ENEMY_NAME[id] or "unknown object",
        status=read(RAM.enemy_state+slot),
        x=read(RAM.enemy_page+slot)*256+read(RAM.enemy_x+slot),
        y=read(RAM.enemy_y+slot)+24,vx=s8(read(0x0058+slot))/16}
    end
  end
  if read(0x0014)==1 and read(0x001B)==0x2E then
    local item_x=math.floor(x/256)*256+read(0x008C)
    if item_x<x-128 then item_x=item_x+256 elseif item_x>x+128 then item_x=item_x-256 end
    state.items[1]={kind="powerup",type=read(0x0039),x=item_x,
      y=read(0x03BE)+16,heading=read(0x004B)}
  end
  return state
end

local function terrain(state)
  local p=state
  local result={gap=nil,landing=nil,obstacle=nil,ceiling=false,grounded=false,
    ground_behind=solid_at(p,p.x-8,p.y+5)}
  local support=solid_at(p,p.x+8,p.y+5)
  result.grounded=support==true and p.vy>=0 and p.vy<=1
  local gap=false
  for dx=12,152,8 do
    local floor=solid_at(p,p.x+dx,p.y+5)
    if floor==false and not gap then result.gap=dx;gap=true
    elseif floor==true and gap then result.landing=dx;break end
  end
  for dx=12,64,8 do
    if solid_at(p,p.x+dx,p.y-10) then result.obstacle=dx;break end
  end
  for dx=0,24,8 do
    if solid_at(p,p.x+dx,p.y-30) then result.ceiling=true;break end
  end
  return result
end

local function nearest_threat(state)
  local selected,selected_time
  for _,enemy in ipairs(state.enemies) do
    local dx=enemy.x-state.x
    local active=not STOMPED[enemy.status]
    if active and dx>-112 and dx<144 and math.abs(enemy.y-state.y)<72 then
      local closing=dx>=0 and state.vx-enemy.vx or enemy.vx-state.vx
      local frames=closing>0 and math.max(0,(math.abs(dx)-16)/closing) or 1000
      if not selected_time or frames<selected_time then selected,selected_time=enemy,frames end
    end
  end
  return selected,selected_time
end

local function valuable_powerup(state)
  for _,item in ipairs(state.items) do
    if (item.type==0 and state.size==1) or (item.type==1 and state.power~=2) or item.type==2 or item.type==3 then
      return item
    end
  end
end

function Bot.new()
  return {frame=0,last_x=nil,still=0,recovery=0,jump_left=0,jump_active=false,
    phase_frames=0,last_reason="starting",last_decision=nil,failed={},
    item_target=nil,item_frames=0,
    abandoned_items={}}
end

local function no_input(reason)
  return {reason=reason}
end

local function signature(state,t,enemy)
  local e="none"
  if enemy then e=tostring(enemy.id)..":"..math.floor(math.abs(enemy.x-state.x)/24) end
  local gap=t.gap and math.floor(t.gap/16) or "clear"
  local obstacle=t.obstacle and math.floor(t.obstacle/16) or "clear"
  return table.concat({math.floor(state.x/16),math.floor(state.y/16),
    state.player_state,gap,obstacle,e},"|")
end

local function simulate(state, candidate, horizon)
  local x,y,vx,vy=state.x,state.y,state.vx,state.vy
  local grounded=state.grounded
  local result={x=x,y=y,risk=0,landed=false}
  for frame=1,horizon do
    local direction=(candidate.right and 1 or 0)-(candidate.left and 1 or 0)
    local cap=candidate.B and 2.5 or 1.7
    if direction~=0 then vx=clamp(vx+direction*0.14,-cap,cap)
    else vx=vx*0.78 end
    if frame==1 and candidate.hold and grounded then
      vy=-5.4;grounded=false
    end
    x=x+vx
    if grounded and solid_at(state,x+8,y+5)==false then
      grounded=false;vy=0
    end
    if not grounded then
      local gravity=frame<=(candidate.hold or 0) and 0.30 or 0.40
      local old_y=y
      vy=math.min(5,vy+gravity);y=y+vy
      if vy>0 and solid_at(state,x+8,y+4)==true then
        grounded=true;result.landed=true;vy=0
        -- Approximate contact by the first supporting tile boundary.
        y=math.floor((y+4)/16)*16-4
      elseif y>state.y+42 then
        local floor=solid_at(state,x+8,y+4)
        if floor==false then result.risk=result.risk+100
        elseif floor==nil then result.risk=result.risk+20 end
      end
      if y<old_y and solid_at(state,x+8,y-10)==true then result.risk=result.risk+80 end
    elseif solid_at(state,x+8,y-10)==true then
      result.risk=result.risk+45
    end
    for _,enemy in ipairs(state.enemies) do
      if not STOMPED[enemy.status] then
        local ex=enemy.x+enemy.vx*frame
        local facing_enemy=(enemy.x<state.x and candidate.left) or
          (enemy.x>state.x and candidate.right)
        local shot_expected=candidate.fire and facing_enemy and frame>=5 and
          math.abs(enemy.y-state.y)<24
        if not shot_expected and math.abs(x-ex)<16 and math.abs(y-enemy.y)<20 then
          local safe_stomp=candidate.hold and vy>0 and y<enemy.y-4
          if not safe_stomp then result.risk=result.risk+65 end
        end
      end
    end
  end
  result.x,result.y,result.vx,result.vy=x,y,vx,vy
  return result
end

local function action_candidates(state,t,enemy,item,bot,sig)
  local candidates={{name="run",right=true,B=true}}
  local gap_has_landing=not t.gap or t.landing~=nil
  if t.grounded and not t.ceiling and gap_has_landing then
    candidates[#candidates+1]={name="short_jump",right=true,B=true,hold=9}
    candidates[#candidates+1]={name="long_jump",right=true,B=true,hold=22}
    candidates[#candidates+1]={name="standing_jump",right=false,B=false,hold=18}
  end
  candidates[#candidates+1]={name="brake"}
  if t.ground_behind==true then candidates[#candidates+1]={name="retreat",left=true,B=true} end
  if state.power==2 and enemy then
    candidates[#candidates+1]={name="fire_forward",right=true,B=true,fire=true}
    if t.ground_behind==true then candidates[#candidates+1]={name="fire_backward",left=true,B=true,fire=true} end
  end
  if item then
    local direction=item.x>=state.x and 1 or -1
    candidates[#candidates+1]={name="seek_powerup",right=direction>0,left=direction<0,B=true,
      objective_x=item.x,reward=item.type==2 and 80 or (item.type==0 and state.size==1 and 72 or 54)}
  end
  local selected,selected_score
  for _,candidate in ipairs(candidates) do
    local predicted=simulate(state,candidate,20)
    local score=predicted.x-state.x-predicted.risk*3
    if candidate.objective_x then
      score=math.abs(candidate.objective_x-state.x)-math.abs(candidate.objective_x-predicted.x)
        +candidate.reward-predicted.risk*3
    end
    if enemy and candidate.fire then
      score=score+math.max(0,42-math.abs(enemy.x-state.x))*0.55
      if candidate.name=="fire_backward" and enemy.vx>state.vx and math.abs(enemy.x-state.x)<64 then
        score=score+132-math.abs(enemy.x-state.x)*0.35
      end
    end
    if t.gap and t.gap<56 and not candidate.hold and candidate.name~="retreat" then score=score-75 end
    if t.gap and t.gap<56 and candidate.hold and not t.landing then score=score-80 end
    if t.obstacle and t.obstacle<44 and candidate.hold then score=score+18 end
    if enemy and enemy.x>state.x and candidate.hold and not STOMPED[enemy.status] then score=score+14 end
    if candidate.name=="short_jump" or candidate.name=="long_jump" or candidate.name=="standing_jump" then
      if not t.gap and not t.obstacle and not (enemy and enemy.x>state.x) then score=score-7 end
    end
    local failed=bot.failed[sig] and bot.failed[sig][candidate.name] or 0
    score=score-failed*90
    if predicted.risk>=60 then score=-math.huge end
    if not selected_score or score>selected_score then selected,selected_score=candidate,score end
  end
  if not selected or selected_score==-math.huge then
    return {name="no_safe_action",reason="no safe predicted action"},selected_score
  end
  selected.reason=selected.name
  return selected,selected_score
end

function Bot.decide(bot,state)
  bot.frame=state.frame
  if state.phase~="playing" then
    bot.phase_frames=bot.phase_frames+1
    bot.jump_left=0;bot.jump_active=false
    if state.phase=="title" and state.frame%90==1 then return {start=true,reason="start game"} end
    return no_input(state.phase)
  end
  bot.phase_frames=0
  local t=terrain(state);state.grounded=t.grounded
  local enemy=nearest_threat(state)
  local item=valuable_powerup(state)
  if item then
    local key=tostring(item.type)..":"..math.floor(item.x/16)
    if bot.item_target~=key then bot.item_target=key;bot.item_frames=0 end
    bot.item_frames=bot.item_frames+1
    if bot.item_frames>90 then bot.abandoned_items[key]=true;item=nil end
    if bot.abandoned_items[key] then item=nil end
  else
    bot.item_target=nil;bot.item_frames=0
  end
  local sig=signature(state,t,enemy)
  local action=action_candidates(state,t,enemy,item,bot,sig)

  -- A no-progress action is remembered for this local state. The next visit
  -- scores different actions higher instead of replaying the failed output.
  if bot.last_x and math.abs(state.x-bot.last_x)<1 and t.grounded then bot.still=bot.still+1
  elseif bot.last_x and state.x>bot.last_x+3 then
    bot.still=0;bot.recovery=0
  else bot.still=math.max(0,bot.still-1) end
  bot.last_x=state.x
  if bot.still>=42 then
    bot.recovery=bot.recovery+1
    local failed=bot.failed[sig] or {}
    local failed_name=bot.last_decision or "run"
    failed[failed_name]=(failed[failed_name] or 0)+1
    bot.failed[sig]=failed
    bot.still=0;bot.jump_left=0;bot.jump_active=false
    -- Re-evaluate after recording the failure so the next output differs.
    action=action_candidates(state,t,enemy,item,bot,sig)
  end
  if action.hold and t.grounded and bot.jump_left==0 and not bot.jump_active then
    bot.jump_left=clamp(action.hold,4,26);bot.jump_active=true
  end
  if bot.jump_left>0 then action.A=true;bot.jump_left=bot.jump_left-1
  else bot.jump_active=false;action.A=nil end
  action.reason=(bot.recovery>0 and "recovery "..bot.recovery..": " or "")..tostring(action.reason or "run")
  bot.last_reason=action.reason
  bot.last_decision=action.name
  return action
end

function Bot.run()
  assert(memory and memory.readbyte and joypad and joypad.set and emu and emu.frameadvance,
    "Load Mario AI Heaven in FCEUX with an NES SMB1 ROM open")
  local bot=Bot.new()
  while true do
    local state=Bot.observe(bot.frame+1)
    local action=Bot.decide(bot,state)
    local buttons={}
    for _,name in ipairs({"left","right","up","down","A","B","start","select"}) do
      if action[name] then buttons[name]=true end
    end
    joypad.set(1,buttons)
    if gui and gui.text then
      gui.text(8,8,"MARIO AI HEAVEN | "..tostring(action.reason or "idle"),"white","black")
      local enemy=nearest_threat(state)
      gui.text(8,18,string.format("x:%d enemy:%s recovery:%d",state.x,
        (enemy and enemy.name) or "none",bot.recovery),"white","black")
    end
    emu.frameadvance()
  end
end

if rawget(_G,"MARIO_AI_TEST") then return Bot end
Bot.run()
