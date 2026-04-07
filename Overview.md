# Dreams: An Overview
By: ME! a solo dev

### 1. Core Identity & Gameplay Loop

##### Genre: 2D Singleplayer Top-Down Procedural Survival RPG (Rust/Bevy)

##### Core Gameplay:
This is a 2D Singleplayer Top-Down Procedural Survival RPG, where the player explores an open, procedurally generated world, engaging in real-time combat, gathering resources, completing quests, and interacting with NPCs, and uncover the secrets of this world. Each in-game day lasts approximately 30 minutes and ends at Midnight, where the entire world resets with a new procedural seed.

##### The Loop:
The world is a fractured mind attempting to reconstruct reality. Each reset is the brain trying to “solve” itself. The player, unknowingly acting as a disruptive force (“the Virus”), causes this process to fail every cycle.

Player progression persists across loops (gear, stats, items), while the world, NPC placements, and terrain regenerate. Key structures and story elements are guaranteed to reappear in different forms each cycle.

##### Short-Term Player Motivation:
Each cycle encourages exploration, resource gathering, quest completion, and character strengthening under the belief that the player is progressing toward saving the world.

---

### 2. Systems & Mechanics

#### Progression (The “Virus” System):
The player collects weapons, food, and survival tools, alongside unique “trauma fragments” tied to story progression. While these function like traditional RPG upgrades, they represent pieces of a traumatic memory.

As the player becomes stronger:

* The world becomes more unstable and distorted
* NPC dialogue becomes less coherent or more fragmented
* Environments may generate in harsher or more broken forms
* Systems subtly degrade, reflecting the “locking in” of trauma

This creates a hidden cost to progression: power reinforces the broken state of the mind.


#### Core Emotion System:

A central “Core Emotion” (e.g., Anger, Sadness, etc.) influences each world reset.

It affects:

* World generation (layout tone, density, hostility)
* NPC behavior and dialogue tone
* Enemy aggression and environmental conditions

Players can influence the Core Emotion through actions and interactions, indirectly shaping the next cycle’s world state.


#### NPC Interaction System:

NPCs are fragmented memories or immune responses within the mind.

* Interaction is based on free-form typed input
* The system parses keywords and emotional tone
* NPCs generate responses dynamically using keywords and contextual phrase pools

Early interactions subtly guide players toward effective communication patterns.

NPC behavior is influenced by the Core Emotion, resulting in varied, sometimes inconsistent responses.

---

### 3. Narrative Structure (The “Hero” Delusion)

#### The Player:
The player believes they are the hero attempting to fix a broken world.

In reality, they are a memory of a real-life attacker—the source of the dreamer’s trauma.

This truth is not revealed immediately. Instead:

* NPC reactions feel subtly “off”
* Acts of “help” may create unease or unclear consequences
* Dialogue contains contradictions that are easy to dismiss early on

The game maintains the illusion of heroism while gradually introducing discomfort and inconsistency.


#### The Girl (Guide Character):

A sarcastic, abrasive guide who introduces mechanics and systems.

* Represents the dreamer’s self-loathing
* Frequently insults or undermines the player
* Her dialogue contains the most direct truths in the game

Her role is to:

* Deliver tutorials
* Provide commentary that players initially dismiss
* Act as a subtle narrative anchor for the underlying reality

---

### 4. The Final Confrontation

#### The Mirror Boss:

The final boss is a “clean” or “static” version of the player character.

* Represents the true, uncorrupted memory of the attacker
* Uses the player’s accumulated gear and stats
* Scaled to be stronger, reflecting the full weight of progression

This transforms the fight into a confrontation with everything the player has built.

The battle is not just mechanical difficulty—it represents the consequence of reinforcing the trauma throughout the game.

---

### 5. Endings

#### Victory Ending:
The player defeats the “static” version.

* The mind’s last defense is destroyed
* The world collapses into a false “utopia”
* The dreamer either dies or remains permanently trapped

The player “wins,” but at the ultimate cost.


#### Sacrifice Ending:
The player allows the static version to erase them.

* The “Virus” is removed
* The world stabilizes
* The system successfully resolves
* The dreamer wakes up

All player progress is deleted as part of the resolution.

---

### 6. Technical Overview

* Engine: Bevy (Rust)
* World Generation:

  * Terrain via Perlin Noise
  * Structures via Wave Function Collapse (WFC)
* Persistent Data:

  * Player stats, inventory, and progression persist across cycles
  * World resets each loop with deterministic structural elements for narrative continuity

* TO BE UPDATED
---

### 7. Current Progress

* 1 month into devolopment working on a MVP
* full single biome (forest) noised, with a player complete and full camera complete
* combat and collision is being worked on right now
* NPC dialog system has a 1-NPC MVP version that read limited keyword and creates dynamic response
* there a fully functional pause menu and loading screen because why not
* temp web demo is being worked on, slight dependency issues with some crate(s)
* TO BE UPDATED
