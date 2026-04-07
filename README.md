# Dreams

This is a 2D Singleplayer Top-Down Procedural Survival RPG, where the player explores an open, procedurally generated world, engaging in real-time combat, gathering resources, completing quests, and interacting with NPCs, and uncover the secrets of this world. Each in-game day lasts approximately 30 minutes and ends at Midnight, where the entire world resets with a new procedural seed.

Overview.md has more info on this and my current progress, don't ask why I've organized it like this because I have no idea either. Note that Overview.md contains basically my entire outline for my game so reading everything there will significantly impact your gameplay experience.

This game is still in the early developmental phrases and therefore please do not judge the game based on what you see now. However feedbacks are welcome, make a pull requewst, report a issue, or DM me in Discord, my usename is *oz_12345_*

Have a nice day~

## Running on the Web

Follow the instructions in wasm.md to run this locally on the web, however it is not recommended because there is significant downgrade in quality and the camera is especially laggy. Most of the errors is mostly likely due to compiling differences than was created when the files was converted into Python.

## TODO/Issues

### MAP 
#### Map bitmasking sometimes fails
(grass example)

tr

mc tr

mc tc tr

mc mc mc tr

the tc above should have been a mc

#### Map water-grass connection
right now when water is beside grass both generates the border, grass should instead generate grass_mc and connect with water_grass.

#### Map doesn't despwn entities
map doesn't despawn entities that out of render distance

#### Tree sprite getting cut off
the top of the big trees doesn't generate and is cut off
the top and right of small tree doesn't generate and is cut off

#### Boulders and Large Bushes are spawning together as a bundle
the bottom two boulder sprites and the top two large bush sprite are being spawned togther as a bundle for some reason

#### Tree placement errors
tree in water, tree too close (on top of each other), in very small patches

#### Decors placement via blue noise
right now the decors of the map is spawning in rigid grids, needs to fixed with more detailed placement via blue noise

### Player 
#### Player jump is animation only
when the player jump w/ space bar it's a animation that plays but nothing really happens

#### Player z-sorting logic is not working
the z-sorting for the player is not work, most likely the map decors are loading on the wrong z-layer

### Other 
#### NPC & Pathfinding
both folders are place holders, empty at the moment

#### Wasm* -unknown -unknown is not supported
wasm* -unknown -unknown is not supported due to a issue with getrandom v0.2.17 js features. this is going to be worked on but isn't likely to be fixed soon

## Credits

TODO
