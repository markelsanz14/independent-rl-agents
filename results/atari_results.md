### Algorithm implementation performance on Atari games:
We compare the performance of the algorithms as reported in the dueling dqn paper ([wang et al. 2015](https://arxiv.org/pdf/1511.06581.pdf)) with ours for the different Atari Games.

This table shows the original performance vs ours implementation. Our results were collected with a replay buffer size of 1M.

| Game                | DQN         |               | Double DQN |     | Double Dueling DQN |              |
|---------------------|-------------|---------------|------------|-----|--------------|--------------------|
|                     | Paper       | Ours (TF)     | Paper     | Ours | Paper        | Paper              |
| Alien               | 1,620       | **3,240**     | 3,747     |      | 4,461              | |
| Amidar              | **978**     | 301           | 1,793     |      | 2,354              | |
| Assault             | **4,280**   | 1,582         | 5,393     |      | 4,621              | |
| Asterix             | 4,359       | **5,380**     | 17,356    |      | 28,188             | |
| Atlantis            | 279,987     | **2,883,892** | 106,056   |      | 382,572            | |
| Bank Heist          | 455         | **926**       | 1,030     |      | 1,611              | |
| Battle Zone         | **29,900**  | 23,650        | 31,700    |      | 37,150             | |
| Beam Rider          | 8,627       | **12,681**    | 13,772    |      | 12,164             | |
| Berzerk             | 585         | **1,362**     | 1,225     |      | 1,472              | |
| Bowling             | **50**      | 33.6          | 68.1      |      | 65.5               | |
| Boxing              | 88          | **98.3**      | 91.6      |      | 99.4               | |
| Breakout            | **385**     | 219           | 418       |      | 345                | |
| Centipede           | 4,657       |            | 5,409     |      | 7,561              | |
| Chopper Command     | 6,126       |            | 5,809     |      | 11,215             | |
| Crazy Climber       | 110,763     |            | 117,282   |      | 143,570            | |
| Demon Attack        | 12,149      |            | 58,044    |      | 60,813             | |
| Double Dunk         | -6.6        | **-3.6**      | -5.5      |      | 0.1                | |
| Enduro              | 729         | **1,770**     | 1,211     |      | 2,258              | |
| Fishing Derby       | -4.9        | **44.1**      | 15.5      |      | 46.4               | |
| Freeway             | 30.8        | **33.6**      | 33.3      |      | 0.0                | |
| Frostbite           | **797**     | 295.1         | 1,683     |      | 4,672              | |
| Gopher              | 8,777       | **18,635**    | 14,840    |      | 15,718             | |
| Gravitar            | 473         | **747**       | 412       |      | 588                | |
| HERO                | **20,437**  | 13,602        | 20,130    |      | 20,818             | |
| Ice Hockey          | **-1.9**    | -3.1          | -2.7      |      | 0.5                | |
| James Bond          | 768         | **5,165**     | 1,358     |      | 1,312              | |
| Kangaroo            | **7,259**   | 4,410         | 12,992    |      | 14,854             | |
| Krull               | **8,422**   | 8,209         | 7,920     |      | 11,451             | |
| Kung-Fu Master      | 26,059      | **29,424**    | 29,710    |      | 34,294             | |
| Montezuma's Revenge | 0.0         | 0.0           | 0.0       |      | 0.0                | |
| Ms Pac-Man          | 3,085       | **3,342**     | 2,711     |      | 6,283              | |
| Name This Game      | 8,207       | **9,294**     | 10,616    |      | 11,971             | |
| Phoenix             | 8,485       | **16,491**    | 12,252    |      | 23,092             | |
| Pitfall!            | -286        | **-37.78**    | -29.9     |      | 0.0                | |
| Pong                | 19.5        | **20.7**      | 20.9      |      | 21.0               | |
| Private Eye         | 146         | **183**       | 129       |      | 103                | |
| Q*Bert              | 13,117      | **15,834**    | 15,088    |      | 19,220             | |
| River Raid          | 7,377       | **21,053**    | 14,884    |      | 21,162             | |
| Road Runner         | 39,544      | **52,086**    | 44,127    |      | 69,524             | |
| Robotank            | **63.9**    | 61.1          | 65.1      |      | 65.3               | |
| Seaquest            | 5,860       | **33,141**    | 16,452    |      | 50,254             | |
| Skiing              | -13,062     | -17,476       | -9,021    |      | -8,857             | |
| Solaris             | 3,482       |            | 3,067     |      | 2,250              | |
| Space Invaders      | 1,692       | **1,974**     | 2,525     |      | 6,427              | |
| Star Gunner         | 54,282      | **63,604**    | 60,142    |      | 89,238             | |
| Tennis              | 12.2        | **23.0**      | -22.8     |      | 5.1                | |
| Time Pilot          | 4,870       | **7,063**     | 8,339     |      | 11,666             | |
| Tutankham           | 68.1        |            | 218       |      | 211                | |
| Up and Down         | 9,989       |            | 22,972    |      | 44,939             | |
| Venture             | 163         |            | 98.0      |      | 497                | |
| Video Pinball       | 196,760     |            | 309,941   |      | 98,209             | |
| Wizard Of Wor       | 2,704       |            | 7,492     |      | 7,855              | |
| Yars' Revenge       | 18,098      |            | 11,712    |      | 49,622             | |
| Zaxxon              | 5,363       |            | 10,163    |      | 12,944             | |
