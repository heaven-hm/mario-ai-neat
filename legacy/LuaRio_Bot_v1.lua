--[[
Luario BOT. Super Mario Bros 1 Bot in LUA. (BETA V 1.0)
will work for Super Mario Bros 2 (Japanese) aka Lost Levels.
Written by Haseeb Mir (haseebmir.hm@gmail.com)


Thanks to SethBlingf for his Enemy and Player Distance Calculation Mehtods used in MarI/O.
getEnemySprites()getPlayerPosition() and other Calculation used are done by him . All credit goes to him.

All the data used in here were taken by this site : https://gist.github.com/1wErt3r/4048722.

Thanks to doppelganger for his awesome SUPER MARIO BROS. DISASSEMBLY guide.
A COMPREHENSIVE SUPER MARIO BROS. DISASSEMBLY by doppelganger (doppelheathen@gmail.com)
]]--


--------------------------------
--Player's Inputs Mapping Table.
--------------------------------
sprint_jump = {right=true,B=true,A=true}
sprint_right = {right=true,B=true}
sprint_left = {left=true,B=true}
sprint_fire_right = {right=true,B=true}
sprint_fire_left = {left=true,B=true}
clear_input = {right=false,A=false,B=false}
jump = {A=true}
fire = {B=true}
right = {right=true}
left = {left=true}
down = {down=true}

------------------------------
--FlagpoleCollision detection.
------------------------------
FlagpoleFNum_YMFDummy = 0x010E
FlagpoleCollisionYPos = 0x070F

---------------------------
--Defining Colors Constants.
---------------------------
backGroundColorRGB = "#5C94FC"
foreGroundColorRGB = "#000000"

-----------------
--Player's States.
-----------------
FireMario = false
SuperMario = false
SmallMario = false

--------------------------
--Bool to Check Direction.
--------------------------
continue_forward = true

-------------------
--Player's Velocity.
-------------------
velocity = 0x009F


-------------------
--Setting Velocity.
-------------------
function setVelocity(speed)
    memory.writebyte(velocity, speed)
end

-----------------------------
--Delays for n amount of time.
-----------------------------
function delayFrameInterval(n)
    local t = os.clock()
    while os.clock() - t <= n do
        
		--Dont Skip Printing.
        printBotName()
        printEnemyNameAndState()
        
        -- skips frame.
        emu.frameadvance()
    end
end


------------------------
--Enemy object constants
------------------------
GreenKoopa = 0x0
BuzzyBeetle = 0x2
RedKoopa = 0x3
HammerBro = 0x5
Goomba = 0x6
Bloober = 0x7
BulletBill_FrenzyVar = 0x8
StaticParatroopa = 0x09
GreyCheepCheep = 0x0a
RedCheepCheep = 0x0b
Podoboo = 0x0c
PiranhaPlant = 0x0d
GreenParatroopaJump = 0x0e
RedParatroopa = 0x0f
GreenParatroopaFly = 0x10
Lakitu = 0x11
Spiny = 0x12
FlyCheepCheepFrenzy = 0x14
FlyingCheepCheep = 0x14
BowserFlame = 0x15
Fireworks = 0x16
BBill_CCheep_Frenzy = 0x17
Stop_Frenzy = 0x18
Bowser = 0x2d
PowerUpObject = 0x2e
VineObject = 0x2f
FlagpoleFlagObject = 0x30
StarFlagObject = 0x31
JumpspringObject = 0x32
BulletBill_CannonVar = 0x33
RetainerObject = 0x35
UpLift			 = 0x26
DownLift = 0x27
Half_lift_Up		 = 0x2B
Half_lift_Down = 0x2C

-----------------------
--Names of All Enemies.
-----------------------
EnemyNameslist = {}
EnemyNameslist[GreenKoopa] = "Koopa"
EnemyNameslist[BuzzyBeetle] = "BuzzyBeetle"
EnemyNameslist[RedKoopa] = "Koopa"
EnemyNameslist[HammerBro] = "HammerBro"
EnemyNameslist[Goomba] = "Goomba"
EnemyNameslist[Bloober] = "Bloober"
EnemyNameslist[BulletBill_FrenzyVar] = "BulletBill"
EnemyNameslist[StaticParatroopa] = "Paratroopa"
EnemyNameslist[GreyCheepCheep] = "CheepCheep"
EnemyNameslist[RedCheepCheep] = "CheepCheep"
EnemyNameslist[Podoboo] = "Podoboo"
EnemyNameslist[PiranhaPlant] = "Piranha Plant"
EnemyNameslist[GreenParatroopaJump] = "Paratroopa Jump"
EnemyNameslist[RedParatroopa] = "Paratroopa"
EnemyNameslist[GreenParatroopaFly] = "Paratroopa Fly"
EnemyNameslist[Lakitu] = "Lakitu"
EnemyNameslist[Spiny] = "Spiny"
EnemyNameslist[FlyCheepCheepFrenzy] = "Fly CheepCheep"
EnemyNameslist[FlyingCheepCheep] = "Fly CheepCheep"
EnemyNameslist[BowserFlame] = "Bowser's Flame"
EnemyNameslist[BBill_CCheep_Frenzy] = "BulletBill"
EnemyNameslist[Bowser] = "Bowser"
EnemyNameslist[BulletBill_CannonVar] = "BulletBill"

------------------------
--Enemy states Constants.
------------------------
alive_enemy = 0x0 --Enemy drawn / Alive.
falling_enemy = 0x01 -- or Bullet_Bill drawn(not killed yet).
enemy_stomped_1 = 0x02 --Goomba Stomped while falling.
enemy_stomped_2 = 0x03 --Goomba Stomped while falling.
enemy_stomped_3 = 0x04 --Goomba Stomped while falling.
spiny_falling = 0x05 --Spiny Falling from Lakitu.
hammerbro_Moving_1 = 0x8 --HammerBro Moving back & forth.
hammerbro_Moving_2 = 0x9 --HammerBro Moving back & forth.
bullet_cheep_hammer_stomped = 0x20 --BulletBill/HammerBro/Cheep stopmed or Real Bowser Killed
killed_with_fire_star = 0x22 --Killed with FireBall or by StarMan.
fakeBowser_killed = 0x23 --FakeBowser Appears in world 1 to world 7.
koopa_stomped_falling = 0xC4 --Koopa Stomped while falling from Paratroopa state.
koopa_stomped_moving_upsideDown = 0x83 --Koopa Moving upside down.
koopa_buzzyBeetle_stomped_moving = 0x84 --Koopa or BuzzyBeetle stopmed and pushed.

----------------------------
--Names of All Enemies State.
----------------------------
EnemyStatelist = {}
EnemyStatelist[alive_enemy] = "Alive"
EnemyStatelist[falling_enemy] = "Falling"
EnemyStatelist[enemy_stomped_1] = "Stomped"
EnemyStatelist[enemy_stomped_2] = "Stomped"
EnemyStatelist[enemy_stomped_3] = "Stomped"
EnemyStatelist[spiny_falling] = "Falling"
EnemyStatelist[hammerbro_Moving_1] = "Moving"
EnemyStatelist[hammerbro_Moving_2] = "Moving"
EnemyStatelist[bullet_cheep_hammer_stomped] = "Stomped"
EnemyStatelist[killed_with_fire_star] = "Killed"
EnemyStatelist[fakeBowser_killed] = "Fake Bowser Killed"
EnemyStatelist[koopa_stomped_falling] = "Stomped and falling"
EnemyStatelist[koopa_stomped_moving_upsideDown] = "Stomped and Moving"
EnemyStatelist[koopa_buzzyBeetle_stomped_moving] = "Stomped and Moving"

-------------------------
--Checking State of Enemy.
-------------------------
function checkEnemyState(_eState)
    
    enemy_state_slot = 0x001E -- 5x Enemy States Range from (0x001E to 0x0023)
    local enemy_state = memory.readbyte(enemy_state_slot + _eState)
    
    --C-Type Ternary Operators.
    return
	   (enemy_state == alive_enemy) and alive_enemy
    or (enemy_state == falling_enemy) and falling_enemy
    or (enemy_state == enemy_stomped_1) and enemy_stomped_1
    or (enemy_state == enemy_stomped_2) and enemy_stomped_2
    or (enemy_state == enemy_stomped_3) and enemy_stomped_3
    or (enemy_state == spiny_falling ) and spiny_falling
    or (enemy_state == hammerbro_Moving_1 ) and hammerbro_Moving_1
    or (enemy_state == hammerbro_Moving_2) and hammerbro_Moving_2
    or (enemy_state == bullet_cheep_hammer_stomped) and bullet_cheep_hammer_stomped
    or (enemy_state == killed_with_fire_star) and killed_with_fire_star
    or (enemy_state == fakeBowser_killed) and fakeBowser_killed
    or (enemy_state == koopa_stomped_falling) and koopa_stomped_falling
    or (enemy_state == koopa_buzzyBeetle_stomped_moving) and koopa_buzzyBeetle_stomped_moving
end

----------------------------------------
--Get EnemyStateName from EnemyStatelist.
----------------------------------------
function getEnemyStateName(enemy_state)
    
    return EnemyStatelist[enemy_state]
    
end

-------------------------
--Checking Type of Enemy.
-------------------------
function checkEnemyID(_enemy_id)
    
    local Enemy_ID = memory.readbyte(0x0016 + _enemy_id)
    
    --C-Type Ternary Operators.
    return
	   (Enemy_ID == Goomba) and Goomba
    or (Enemy_ID == GreenKoopa) and GreenKoopa
    or (Enemy_ID == RedKoopa) and RedKoopa
    or (Enemy_ID == BuzzyBeetle) and BuzzyBeetle
    or (Enemy_ID == GreenParatroopaFly) and GreenParatroopaFly
    or (Enemy_ID == GreenParatroopaJump) and GreenParatroopaJump
    or (Enemy_ID == RedParatroopa) and RedParatroopa
    or (Enemy_ID == PiranhaPlant) and PiranhaPlant
    or (Enemy_ID == BulletBill_CannonVar) and BulletBill_CannonVar
    or (Enemy_ID == HammerBro) and HammerBro
    or (Enemy_ID == Lakitu) and Lakitu
    or (Enemy_ID == Bowser) and Bowser
    or (Enemy_ID == UpLift) and UpLift
    or (Enemy_ID == DownLift) and DownLift
    or (Enemy_ID == JumpspringObject) and JumpspringObject
    or (Enemy_ID == FlagpoleFlagObject) and FlagpoleFlagObject
end

-----------------------------------
--Get EnemyName from EnemyNameslist.
------------------------------------
function getEnemyName(Enemy_ID)
    
    return EnemyNameslist[Enemy_ID]
    
end


-----------------------------------------------------
--Get Enemy's x and y position if they are drawn yet.
-----------------------------------------------------
function getEnemySprites()
    
    local EnemySprites = {}
    for slot=0,4 do
        local enemy = memory.readbyte(0xF+slot)
        if enemy ~= 0 then
            local ex = memory.readbyte(0x6E + slot)*0x100 + memory.readbyte(0x87+slot)
            local ey = memory.readbyte(0xCF + slot)+24
            EnemySprites[#EnemySprites+1] = {["x"]=ex,["y"]=ey}
        end
    end
    
    return EnemySprites
    
end

---------------------------------------------------
--Get Player's x and y position with screen offset.
---------------------------------------------------
function getPlayerPosition()
    
    marioX = memory.readbyte(0x6D) * 0x100 + memory.readbyte(0x86)
    marioY = memory.readbyte(0x03B8)+16
    
    screenX = memory.readbyte(0x03AD)
    screenY = memory.readbyte(0x03B8)
    
end

---------------------------------------------------
--Checking for Collision with Object Pipe / Brick ?
---------------------------------------------------
function playerObjectCollision()
    
    local playerCollisionBits = memory.readbyte(0x0490)
    local collided = 0xFE
    
    if(playerCollisionBits == collided)then
        return true
    end
    
    --OtherWise.
    return false
end


-----------------------------------
--Checking Current State of player.
-----------------------------------
function checkPlayerState()
    
    --Small Mario : Size = 1, Status = 0
    --Super Mario : Size = 0, Status = 1
    --Fire Mario : Size = 0, Status = 2
    
    local PlayerSize = memory.readbyte(0x0754)
    local PlayerStatus = memory.readbyte(0x0756)
    
    if(PlayerSize == 0x1 and PlayerStatus == 0x0)then
        
        SuperMario = false
        FireMario = false
        
        SmallMario = true
        return SmallMario
        
    end
    
    if(PlayerSize == 0x0 and PlayerStatus == 0x1)then
        
        SmallMario = false
        FireMario = false
        
        SuperMario = true
        return SuperMario
        
    end
    
    if(PlayerSize == 0x0 and PlayerStatus == 0x2)then
        
        SmallMario = false
        SuperMario = false
        
        FireMario = true
        return FireMario
        
    end
    
    
end


---------------------------------
--Checking Miscellaneous Objects.
---------------------------------
function checkMiscObj()

    for misc_slot = 0,4 do
        if(checkEnemyID(misc_slot) == JumpspringObject or checkEnemyID(misc_slot) == StarFlagObject or 
		   checkEnemyID(misc_slot) == FlagpoleFlagObject or checkEnemyID(misc_slot) == RetainerObject or
		   checkEnemyID(misc_slot) == Fireworks) then	  
		
		 return true
        end
    end

	--OtherWise
	return false
end


------------------
--Prints Bot Name.
------------------
function printBotName()
    gui.text(19,16,"LU", "white" , "#5C94FC")
    gui.text(68,16,"BOT", "white" , "#5C94FC")
end

------------------------------
--Prints Enemy Name And State.
------------------------------
function printEnemyNameAndState()
    for slot = 0,4 do --5 Enemies per page.
        
        -------------------
        --Print Bot's Name.
        -------------------
        printBotName()
        
        --------------------------------
        --Defining Re-Used Enemy States.
        --------------------------------
        if(checkEnemyID(slot) == Goomba)then
            EnemyStatelist[falling_enemy] = "Falling"
            
        else
            if(checkEnemyID(slot) == HammerBro)then
                EnemyStatelist[falling_enemy] = "Jumping"
                
            else
                if(checkEnemyID(slot) == BulletBill_CannonVar)then
                    EnemyStatelist[falling_enemy] = "Moving"
                end
            end
        end
        
        if(checkEnemyID(slot) == Bowser)then
            EnemyStatelist[bullet_cheep_hammer_stomped] = "Killed"
        end
        
        
        if(memory.readbyte(0x0F + slot) > 0 )then
            eType = memory.readbyte(0x0016 + slot)
            eName = (getEnemyName(eType) ~= nil or (checkMiscObj() == false)) and getEnemyName(eType) or "No Enemy" --For Miscellaneous/Unknown Objects.
            
			if(eName ~= "No Enemy")then
				state = memory.readbyte(0x001E + slot)
				eState = getEnemyStateName(state) and getEnemyStateName(state) or "Void State" --For Miscellaneous/Unknown States.
            end
			
            if(eState == "Void State")then
                emu.print("Void State : ",state)
            end
            
			else
				--For Empty Enemy Slots.
				eName = "No Enemy"
				eState = "Nil State"
			end
        
		--Prints Enemy name and state.
        gui.text(8, 36 + (slot * 8), string.format("Enemy %d : %s, State : %s",slot + 1,eName,eState),foreGroundColorRGB,backGroundColorRGB)
    end
end


--TODO Skip 18-21 Frames For Next Jump Input Sequence.

----------------------------
--Checking Here for PowerUp.
----------------------------
function checkPowerUp()
    
    --Powerup type (when on screen)
    --0 - Mushroom
    --1 - Flower
    --2 - Star
    --3 - 1up
    
    --PowerUpDrawn
    --0 - No
    --1 - Yes
    
    --[[
    Powerup on screen
    0x00 - No
    0x2E - Yes
    ]]--
    
    --[[
    Shroom heading
    1 - Right
    2 - Left
    ]]--
    
    --.db $47, $47, $47, $47 brick (power-up)
    
    
    local PowerUpType = 0x0039
    local PowerUpObject = 0x002e
    local PowerUpDrawn = 0x0014
    local ShroomHeading = 0x004B
    local Powerup_on_screen = 0x001B
	local shroom_left = 0x02
	local shroom_right = 0x01
    
	
	--------------------------------
	--Checking For Mushroom PowerUp.
	--------------------------------	
    if(memory.readbyte(PowerUpDrawn) == 0x1 and memory.readbyte(PowerUpType) == 0x00 and memory.readbyte(Powerup_on_screen) == 0x2E)then
        
        continue_forward = false
		memory.writebyte(0x0057,0x0) --Set Player's Speed to 0.

        if(memory.readbyte(ShroomHeading) == shroom_right and memory.readbyte(PowerUpDrawn) == 0x1)then
            			
            while(checkPlayerState() ~= SuperMario and memory.readbyte(PowerUpDrawn) == 0x1) do
				
                joypad.set(1,sprint_right)
				emu.frameadvance()
				
				if(memory.readbyte(ShroomHeading) == shroom_left)then
					break
				end
				
				----------------------------------------------------
				--Checking if Enemy is present while taking PowerUp.
				----------------------------------------------------
				if(checkEnemyState(0x0) == alive_enemy or checkEnemyState(0x01) == alive_enemy)then
					joypad.set(1,sprint_jump)
					 
				emu.frameadvance()
				end

            end
            
        else if(memory.readbyte(ShroomHeading) == shroom_left and memory.readbyte(PowerUpDrawn) == 0x1)then
                
                while(checkPlayerState() ~= SuperMario and memory.readbyte(PowerUpDrawn) == 0x1) do
								
                    joypad.set(1,sprint_left)
                    emu.frameadvance()
				
					if(memory.readbyte(ShroomHeading) == shroom_right)then
						break
					end
				
					----------------------------------------------------
					--Checking if Enemy is present while taking PowerUp.
					----------------------------------------------------
					if(checkEnemyState(0x0) == alive_enemy or checkEnemyState(0x01) == alive_enemy)then
						joypad.set(1,sprint_jump)
					 
					 emu.frameadvance()
				 end
				
                end
                
            end
            
        end
    
		else    
        continue_forward = true
    end
	
    
    --TODO Check for other PowerUps also.
end



-----------------------------------------------------------------------------
--Running Several Player/Enemies Collision and distance calculation routines.
-----------------------------------------------------------------------------
function RunGameEngineSubRoutines()
    
    -------------------
    --Print Bot's Name.
    -------------------
    printBotName()
    
    getPlayerPosition()
    EnemySprites = getEnemySprites()
    
    local BoxRadius = 6
    local misc_obj = false
    
    --Checking for Miscellaneous Object.
	if(checkMiscObj() == true)then
		misc_obj = true
	end
    
    --Setting Player's Direction.
    if(continue_forward == true)then
        joypad.set(1,sprint_right)
    end
    
    --Dont Calculate distance on Miscellaneous objects.
    if(not misc_obj)then
        
        for dy=-BoxRadius*16,BoxRadius*16,16 do
            for dx=-BoxRadius*16,BoxRadius*16,16 do
                
                for i = 1,#EnemySprites do
                    
                    --------------------
                    --Check For PowerUp
                    --------------------
					if(checkPlayerState() == SmallMario)then
                     checkPowerUp()
					end
                    
                    ----------------------------------------------------------
                    --Calculating Absolute Distance Between Player and Enemy.
                    ----------------------------------------------------------
                    distx = math.abs(EnemySprites[i]["x"] - (marioX+dx))
                    disty = math.abs(EnemySprites[i]["y"] - (marioY+dy))
                    
                    ---------------------------
                    --for Enemy 5x sprites Loop.
                    ---------------------------
                    for EnemySlotNum = 0,4 do
						
                    ------------------------------------------------------------
                    --Calculating Relative Distance Between Player and Enemy..
                    ------------------------------------------------------------
					if(memory.readbyte(0x0F + EnemySlotNum) > 0)then
						x = (EnemySprites[i]["x"] - (marioX+dx))
						y = (EnemySprites[i]["y"] - (marioY+dy))
											
                        -----------------------------
                        --For Enemies on Left side.
                        -----------------------------
                        if( ((x >= -190 and x <= -111) and (y >= -80 and y <= -20)) and
                        (checkEnemyState(EnemySlotNum) == alive_enemy and checkPlayerState() == FireMario))then
									joypad.set(1,clear_input)
									joypad.set(1,sprint_fire_left)
									
									delayFrameInterval(0.1)			
									joypad.set(1,left)
									joypad.set(1,fire)
                        end
						
						else 
						continue_forward = true
					end	
                        
					
						
                        -----------------------------------------------------
                        --For Regular Enemies do checkEnemy distx and disty.
                        -----------------------------------------------------
                        if(distx <= 8 and disty <= 8)then
                            
                            -------------------------
                            --Printing Enemies names.
                            -------------------------
                            printEnemyNameAndState()
                            
                            -------------------------------------------------
                            --Checking for Flying Enemies or Enemies at Height.
                            -------------------------------------------------
                            if(checkEnemyID(EnemySlotNum) == PiranhaPlant and checkEnemyState(EnemySlotNum) == alive_enemy
                            or checkEnemyID(EnemySlotNum) == RedParatroopa or checkEnemyID(EnemySlotNum) == GreenParatroopaFly or checkEnemyID(EnemySlotNum) == GreenParatroopaJump)
                            or checkEnemyID(EnemySlotNum) == Lakitu
                            then
                                
                                joypad.set(1,sprint_right)
                                joypad.set(1,sprint_jump)
                                setVelocity(0xFC)
                                delayFrameInterval(0.1)
                                joypad.set(1,fire)
                                
                                -----------------------------------------------------------------------------------
                                --Special Condition for PiranhaPlant since the dont get removed from Enemy Page :(
                                -----------------------------------------------------------------------------------
                                if(checkEnemyID(EnemySlotNum) == PiranhaPlant)then --checkEnemyState(EnemySlotNum) == killed_with_fire_star)then
                                    
                                    --Remove PiranhaPlant from this slot explicitly.
                                    memory.writebyte(0x0F + EnemySlotNum,0x0)
                                    memory.writebyte(0x001E + EnemySlotNum,0x0)
                                    
                                    joypad.set(1,sprint_right)
                                    joypad.set(1,sprint_jump)
                                    setVelocity(0xFC)
                                end
                                
                            end--End of Checking for Flying Enemies
                            
                            -------------------------------------------------
                            --Checking for Lift & JumpspringObject (Skip it).
                            -------------------------------------------------
                            if(checkEnemyID(EnemySlotNum) == UpLift or checkEnemyID(EnemySlotNum) == DownLift or checkEnemyID(EnemySlotNum) == JumpspringObject)then
                                joypad.set(1,clear_input)
                                delayFrameInterval(0.1)
                                
                                joypad.set(1,sprint_jump)
                                break
                            end
                            
                            ------------------------------
                            --Checking for falling Enemies.
                            -----------------------------
                            if(checkEnemyState(EnemySlotNum) == falling_enemy)then
                                
                                joypad.set(1,clear_input)
                                delayFrameInterval(0.1)
                                joypad.set(1,jump)
                            end
                            
                            ------------------------------
                            --Checking for Ground Enemies.
                            -----------------------------
                            if(checkEnemyID(EnemySlotNum) == Goomba or checkEnemyID(EnemySlotNum) == GreenKoopa or checkEnemyID(EnemySlotNum) == RedKoopa or checkEnemyID(EnemySlotNum) == HammerBro or checkEnemyID(EnemySlotNum) == BuzzyBeetle or checkEnemyID(EnemySlotNum) == BulletBill_CannonVar or checkEnemyID(EnemySlotNum) == Bowser
                            )then
                                
                                --------------------------------
                                --Checking the PlayerState here.
                                ------------------------------
                                if(checkPlayerState() == SmallMario or checkPlayerState() == SuperMario)then
                                    joypad.set(1,sprint_jump)--For Stomping enemies.
                                    
                                else if(checkPlayerState() == FireMario)then
                                        if(checkEnemyID(EnemySlotNum) == BuzzyBeetle or checkEnemyID(EnemySlotNum) == BulletBill_CannonVar)then
                                            
                                            joypad.set(1,clear_input)
                                            delayFrameInterval(0.1)
                                            joypad.set(1,jump) --Jump on BuzzyBeetle or BulletBill dont throw FireBalls.
                                            
                                        else --For rest ground enemies throw FireBalls.
                                            joypad.set(1,sprint_right)
                                            joypad.set(1,clear_input)
                                            
                                            joypad.set(1,right)
                                            delayFrameInterval(0.1)
                                            joypad.set(1,fire)--Throw Fire Balls.
                                            joypad.set(1,sprint_right)
                                            
                                        end
                                    end
                                end
                            end--End of Checking for Ground Enemies
							
							--Main Frame of Loop unless skipped in FrameInterval.		
                            emu.frameadvance()
							
                        end--End of Enemy 5x Sprites.
                    end
                end
            end
        end
    end
end

---------------------------
--Clearig JoyPad on Death.
---------------------------
function clearJoyPad()

 local OperMode_Task = 0x0772
 
 --Clear Joypad untill loading screen appears.
 while(memory.readbyte(OperMode_Task)~= 0x0) do
	joypad.set(1,clear_input)
	emu.frameadvance()
  end
  
 end

-----------------------------
--Checking if player is dead
-----------------------------
function isPlayerDead()

local deathMusicLoaded = 0x0712
local playerState = 0x000E

 if(memory.readbyte(deathMusicLoaded) == 0x01 or memory.readbyte(playerState) == 0x0B)then
 return true

 else
 return false

 end
end 

-----------------
--Main Game Loop.
-----------------
while(true)do
        
    -------------------
    --Print Bot's Name.
    -------------------
    printBotName()
    
    ---------------------------------
    --	Testing Section Starts.
    --Used for Testing Purpose Only.
    -------------------------------
    
    --memory.writebyte(0x079e,0xFF)--set injury time.
    --memory.writebyte(0x79F,0xFF)--Makes Player Invincible.
    
    --memory.writebyte(0x0754,0x0)--PlayerSize
    --memory.writebyte(0x0756,0x2)--PlayerStatus
    
    --Three Bytes of Timer.
    --memory.writebyte(0x0007F8,0x09)--Increase Time to Max.
    --memory.writebyte(0x0007F9,0x09)--That is
    --memory.writebyte(0x0007FA,0x09)-- 999 Timer.
    
    --memory.writebyte(0x0787,0x2)--Freeze Timer.
    --memory.writebyte(0x075A,0x03)--Set Lives to 10
    -----------------------
    --Testing Section Ends.
    -----------------------
    
    ------------------------
    --Printing Enemies names.
    ------------------------
    printEnemyNameAndState()
    
	-----------------------
    --Clear Input on Death.
    -----------------------
	if(isPlayerDead())then
        clearJoyPad()
    end
	
    -------------------------
    --Checking for Collision.
    -------------------------
    local SpriteHitDetectFlag = 0x0722
    
    if(playerObjectCollision()) then --Collided with Pipe or Block ?
        joypad.set(1,sprint_jump)
        
        setVelocity(-3)
        memory.writebyte(SpriteHitDetectFlag,0xFF)
        
        joypad.set(1,sprint_right)
    end
    
    --------------------------------------------------------------------------------------------------------------------------------
    --For Faster Accelaration (has Bug in some stages it might not complete as Mario has already Moved too Far from its x-Position )
    --------------------------------------------------------------------------------------------------------------------------------
    --[[local PlayerHSpeed = memory.readbyte(0x0057)
    
    if(PlayerHSpeed == 0x0)then
        
        joypad.set(1,{left=true,right=true})
        emu.frameadvance()
        joypad.set(1,sprint_jump)
        emu.frameadvance()
        joypad.set(1,left)
        
    else
        joypad.set(1,sprint_right)
    end]]--
    
    -----------------------------------
    --Dont Input after clearing stage.
    -----------------------------------
    if(memory.readbyte(FlagpoleFNum_YMFDummy) == 0x3E and memory.readbyte(FlagpoleCollisionYPos) == 0xA0)then
        joypad.set(1,clear_input)
         
    else
        -------------------------------------------------
        --Run Several Object and Enemy Detection Routines.
        ------------------------------------------------
        RunGameEngineSubRoutines()
    end
    
	--Advance to Next Frame.
    emu.frameadvance()
end
