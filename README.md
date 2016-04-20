# PuddleDetector

Visual-LIDAR system to detect puddles even in cluttered environments. Existing methods struggle in cluttered environments:

* Most deep-learning methods fail to distinguish between reflections of things like foliage, buildings etc from the foliage and buildings themselves. As such, they are bad at detecting reflective puddles of water, which have no appearance of their own.
* Some LIDAR-based methods detect puddles by using the fact that water attenuates near infrared light significantly, so that puddles often show up as holes in the data. However, this is unreliable and sometimes cannot distinguish between dark-coloured material from puddles.

Our method estimates the normal direction of the ground plane and computes the reflection of surrounding objects using range information from the LIDAR. This is able to avoid the shortcomings of both methods described above.
