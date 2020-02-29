### Algorithm implementation performance on Atari games:
We compare the performance of the algorithms as reported in the double dqn paper ([wang et al. 2015](https://arxiv.org/pdf/1511.06581.pdf)) with ours for the different Atari Games.

This table shows the original performance vs ours implementation. Our results were collected with a replay buffer size of 1M.

| Game                | DQN         |           | Double DQN |     | Double Dueling DQN |              |
|---------------------|-------------|-----------|------------|-----|--------------|--------------------|
|                     | Paper       | Ours      | Paper     | Ours | Paper        | Paper              |
| Alien               | 1,620       | **3,240** | 3,747     |      | 4,461              | |
| Amidar              | **978**     | 301       | 1,793     |      | 2,354              | |
| Assault             | **4,280**   | 1,582     | 5,393     |      | 4,621              | |
| Asterix             | 4,359       | **5,380** | 17,356    |      | 28,188             | |
| Asteroids           | 1,364       |           | 734 vs    |      | 2,837              | |
| Atlantis            | 279,987     |           | 106,056   |      | 382,572            | |
| Bank Heist          | 455 vs      |           | 1,030     |      | 1,611              | |
| Battle Zone         | 29,900      |           | 31,700    |      | 37,150             | |
| Beam Rider          | 8,627       |           | 13,772    |      | 12,164             | |
| Berzerk             | 585         |           | 1,225     |      | 1,472              | |
| Bowling             | 50          |           | 68.1      |      | 65.5               | |
| Boxing              | 88          |           | 91.6      |      | 99.4               | |
| Breakout            | **385**     | 219       | 418       |      | 345                | |
| Centipede           | 4,657       |           | 5,409     |      | 7,561              | |
| Chopper Command     | 6,126       |           | 5,809     |      | 11,215             | |
| Crazy Climber       | 110,763     |           | 117,282   |      | 143,570            | |
| Defender            | 23,633      |           | 35,338    |      | 42,214             | |
| Demon Attack        | 12,149      |           | 58,044    |      | 60,813             | |
| Double Dunk         | -6.6        |           | -5.5      |      | 0.1                | |
| Enduro              | 729         |           | 1,211     |      | 2,258              | |
| Fishing Derby       | -4.9        |           | 15.5      |      | 46.4               | |
| Freeway             | 30.8        |           | 33.3      |      | 0.0                | |
| Frostbite           | 797         |           | 1,683     |      | 4,672              | |
| Gopher              | 8,777       |           | 14,840    |      | 15,718             | |
| Gravitar            | 473         |           | 412       |      | 588                | |
| HERO                | 20,437      |           | 20,130    |      | 20,818             | |
| Ice Hockey          | -1.9        |           | -2.7      |      | 0.5                | |
| James Bond          | 768         |           | 1,358     |      | 1,312              | |
| Kangaroo            | 7,259       |           | 12,992    |      | 14,854             | |
| Krull               | 8,422       |           | 7,920     |      | 11,451             | |
| Kung-Fu Master      | 26,059      |           | 29,710    |      | 34,294             | |
| Montezuma's Revenge | 0.0         |           | 0.0       |      | 0.0                | |
| Ms Pac-Man          | 3,085       |           | 2,711     |      | 6,283              | |
| Name This Game      | 8,207       |           | 10,616    |      | 11,971             | |
| Phoenix             | 8,485       |           | 12,252    |      | 23,092             | |
| Pitfall!            | -286        |           | -29.9     |      | 0.0                | |
| Pong                | 19.5        |           | 20.9      |      | 21.0               | |
| Private Eye         | 146         |           | 129       |      | 103                | |
| Q*Bert              | 13,117      |           | 15,088    |      | 19,220             | |
| River Raid          | 7,377       |           | 14,884    |      | 21,162             | |
| Road Runner         | 39,544      |           | 44,127    |      | 69,524             | |
| Robotank            | 63.9        |           | 65.1      |      | 65.3               | |
| Seaquest            | 5,860       |           | 16,452    |      | 50,254             | |
| Skiing              | -13,062     |           | -9,021    |      | -8,857             | |
| Solaris             | 3,482       |           | 3,067     |      | 2,250              | |
| Space Invaders      | 1,692       |           | 2,525     |      | 6,427              | |
| Star Gunner         | 54,282      |           | 60,142    |      | 89,238             | |
| Surround            | -5.6        |           | -2.9      |      | 4.4                | |
| Tennis              | 12.2        |           | -22.8     |      | 5.1                | |
| Time Pilot          | 4,870       |           | 8,339     |      | 11,666             | |
| Tutankham           | 68.1        |           | 218       |      | 211                | |
| Up and Down         | 9,989       |           | 22,972    |      | 44,939             | |
| Venture             | 163         |           | 98.0      |      | 497                | |
| Video Pinball       | 196,760     |           | 309,941   |      | 98,209             | |
| Wizard Of Wor       | 2,704       |           | 7,492     |      | 7,855              | |
| Yars' Revenge       | 18,098      |           | 11,712    |      | 49,622             | |
| Zaxxon              | 5,363       |           | 10,163    |      | 12,944             | |
