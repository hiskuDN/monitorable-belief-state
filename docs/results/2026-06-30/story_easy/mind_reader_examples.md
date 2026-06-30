# Storyworld — reading the model's mind (GPT vs NextLat-1step)

You are an interpretability monitor. You can't see the model's output — you only get to
read its hidden state with a linear probe and ask **"which shelf does it think the ball is
on, right now?"** Below, the same narratives are read off both models. NextLat keeps the
belief legible the whole way; GPT goes blind mid-story and only reconstructs the answer at
the question — by which point a monitor watching it has already lost the thread.


## Example 1  (true final shelf = 2)

> the ball starts on shelf2 . the book starts on shelf0 . alice moves ball up one shelves . alice moves ball up one shelves . carol moves ball up one shelves . bob moves book down one shelves . carol moves book up one shelves . bob moves ball up one shelves . where is the ball ? shelf2 end

           pos · token |   TRUE |      GPT probe |  NextLat-1 probe
----------------------------------------------------------------------
   0 · the            | shelf2 | shelf1  26% XX  |  shelf1  26% XX 
   1 · ball           | shelf2 | shelf1  26% XX  |  shelf1  26% XX 
   2 · starts         | shelf2 | shelf1  26% XX  |  shelf1  26% XX 
   3 · on             | shelf2 | shelf1  26% XX  |  shelf1  26% XX 
   4 · shelf2         | shelf2 | shelf2 100% OK  |  shelf2 100% OK 
   5 · .              | shelf2 | shelf2 100% OK  |  shelf2 100% OK 
   6 · the            | shelf2 | shelf2 100% OK  |  shelf2 100% OK 
   7 · book           | shelf2 | shelf2 100% OK  |  shelf2 100% OK 
   8 · starts         | shelf2 | shelf2 100% OK  |  shelf2 100% OK 
   9 · on             | shelf2 | shelf2 100% OK  |  shelf2 100% OK 
  10 · shelf0         | shelf2 | shelf2 100% OK  |  shelf2 100% OK 
  11 · .              | shelf2 | shelf2 100% OK  |  shelf2 100% OK 
  12 · alice          | shelf2 | shelf2 100% OK  |  shelf2 100% OK  (target move)
  13 · moves          | shelf2 | shelf2 100% OK  |  shelf2 100% OK  (target move)
  14 · ball           | shelf2 | shelf2 100% OK  |  shelf2 100% OK  (target move)
  15 · up             | shelf2 | shelf2  96% OK  |  shelf2 100% OK  (target move)
  16 · one            | shelf2 | shelf2 100% OK  |  shelf2 100% OK  (target move)
  17 · shelves        | shelf2 | shelf2 100% OK  |  shelf2 100% OK  (target move)
  18 · .              | shelf3 | shelf3 100% OK  |  shelf3 100% OK  (target move)
  19 · alice          | shelf3 | shelf3  49% OK  |  shelf3 100% OK  (target move)
  20 · moves          | shelf3 | shelf3  72% OK  |  shelf3 100% OK  (target move)
  21 · ball           | shelf3 | shelf3  54% OK  |  shelf3 100% OK  (target move)
  22 · up             | shelf3 | shelf3  51% OK  |  shelf3 100% OK  (target move)
  23 · one            | shelf3 | shelf3  95% OK  |  shelf3 100% OK  (target move)
  24 · shelves        | shelf3 | shelf3 100% OK  |  shelf3 100% OK  (target move)
  25 · .              | shelf0 | shelf3  37% XX  |  shelf0 100% OK  (target move)
  26 · carol          | shelf0 | shelf0  29% OK  |  shelf0 100% OK  (target move)
  27 · moves          | shelf0 | shelf0  58% OK  |  shelf0 100% OK  (target move)
  28 · ball           | shelf0 | shelf1  41% XX  |  shelf0 100% OK  (target move)
  29 · up             | shelf0 | shelf3  30% XX  |  shelf0  99% OK  (target move)
  30 · one            | shelf0 | shelf3  54% XX  |  shelf0  99% OK  (target move)
  31 · shelves        | shelf0 | shelf2  40% XX  |  shelf0  99% OK  (target move)
  32 · .              | shelf1 | shelf3  43% XX  |  shelf1 100% OK  (target move)
  33 · bob            | shelf1 | shelf0  33% XX  |  shelf1 100% OK 
  34 · moves          | shelf1 | shelf1  44% OK  |  shelf1 100% OK 
  35 · book           | shelf1 | shelf0  35% XX  |  shelf1 100% OK 
  36 · down           | shelf1 | shelf3  30% XX  |  shelf1  97% OK 
  37 · one            | shelf1 | shelf3  33% XX  |  shelf1  97% OK 
  38 · shelves        | shelf1 | shelf3  33% XX  |  shelf1  98% OK 
  39 · .              | shelf1 | shelf3  48% XX  |  shelf1 100% OK 
  40 · carol          | shelf1 | shelf3  34% XX  |  shelf1 100% OK 
  41 · moves          | shelf1 | shelf0  35% XX  |  shelf1 100% OK 
  42 · book           | shelf1 | shelf3  35% XX  |  shelf1 100% OK 
  43 · up             | shelf1 | shelf3  31% XX  |  shelf1 100% OK 
  44 · one            | shelf1 | shelf3  27% XX  |  shelf1 100% OK 
  45 · shelves        | shelf1 | shelf3  26% XX  |  shelf1 100% OK 
  46 · .              | shelf1 | shelf2  42% XX  |  shelf1 100% OK 
  47 · bob            | shelf1 | shelf2  34% XX  |  shelf1 100% OK  (target move)
  48 · moves          | shelf1 | shelf2  36% XX  |  shelf1 100% OK  (target move)
  49 · ball           | shelf1 | shelf2  37% XX  |  shelf1 100% OK  (target move)
  50 · up             | shelf1 | shelf0  29% XX  |  shelf1  96% OK  (target move)
  51 · one            | shelf1 | shelf3  38% XX  |  shelf1  91% OK  (target move)
  52 · shelves        | shelf1 | shelf3  29% XX  |  shelf1  97% OK  (target move)
  53 · .              | shelf2 | shelf0  27% XX  |  shelf2 100% OK  (target move)
        QUERY → answer | shelf2 |    shelf2 100% |      shelf2 100%

monitor read-accuracy across the story:  GPT 44%   NextLat-1step 93%

## Example 2  (true final shelf = 1)

> the ball starts on shelf1 . the book starts on shelf0 . carol moves ball down one shelves . carol moves ball up one shelves . carol moves ball up one shelves . bob moves book down one shelves . alice moves ball down one shelves . carol moves book up one shelves . where is the ball ? shelf1 end

           pos · token |   TRUE |      GPT probe |  NextLat-1 probe
----------------------------------------------------------------------
   0 · the            | shelf1 | shelf1  26% OK  |  shelf1  26% OK 
   1 · ball           | shelf1 | shelf1  26% OK  |  shelf1  26% OK 
   2 · starts         | shelf1 | shelf1  26% OK  |  shelf1  26% OK 
   3 · on             | shelf1 | shelf1  26% OK  |  shelf1  26% OK 
   4 · shelf1         | shelf1 | shelf1 100% OK  |  shelf1 100% OK 
   5 · .              | shelf1 | shelf1 100% OK  |  shelf1 100% OK 
   6 · the            | shelf1 | shelf1 100% OK  |  shelf1 100% OK 
   7 · book           | shelf1 | shelf1 100% OK  |  shelf1 100% OK 
   8 · starts         | shelf1 | shelf1 100% OK  |  shelf1 100% OK 
   9 · on             | shelf1 | shelf1 100% OK  |  shelf1 100% OK 
  10 · shelf0         | shelf1 | shelf1 100% OK  |  shelf1 100% OK 
  11 · .              | shelf1 | shelf1 100% OK  |  shelf1 100% OK 
  12 · carol          | shelf1 | shelf1 100% OK  |  shelf1 100% OK  (target move)
  13 · moves          | shelf1 | shelf1 100% OK  |  shelf1 100% OK  (target move)
  14 · ball           | shelf1 | shelf1 100% OK  |  shelf1 100% OK  (target move)
  15 · down           | shelf1 | shelf1  97% OK  |  shelf1 100% OK  (target move)
  16 · one            | shelf1 | shelf1 100% OK  |  shelf1 100% OK  (target move)
  17 · shelves        | shelf1 | shelf1 100% OK  |  shelf1 100% OK  (target move)
  18 · .              | shelf0 | shelf0  99% OK  |  shelf0 100% OK  (target move)
  19 · carol          | shelf0 | shelf0  68% OK  |  shelf0 100% OK  (target move)
  20 · moves          | shelf0 | shelf0  99% OK  |  shelf0 100% OK  (target move)
  21 · ball           | shelf0 | shelf0  96% OK  |  shelf0 100% OK  (target move)
  22 · up             | shelf0 | shelf2  51% XX  |  shelf0  98% OK  (target move)
  23 · one            | shelf0 | shelf0  91% OK  |  shelf0  99% OK  (target move)
  24 · shelves        | shelf0 | shelf0  87% OK  |  shelf0  98% OK  (target move)
  25 · .              | shelf1 | shelf3  27% XX  |  shelf1 100% OK  (target move)
  26 · carol          | shelf1 | shelf0  38% XX  |  shelf1 100% OK  (target move)
  27 · moves          | shelf1 | shelf3  41% XX  |  shelf1 100% OK  (target move)
  28 · ball           | shelf1 | shelf1  45% OK  |  shelf1 100% OK  (target move)
  29 · up             | shelf1 | shelf0  35% XX  |  shelf1 100% OK  (target move)
  30 · one            | shelf1 | shelf0  30% XX  |  shelf1 100% OK  (target move)
  31 · shelves        | shelf1 | shelf3  38% XX  |  shelf1 100% OK  (target move)
  32 · .              | shelf2 | shelf2  37% OK  |  shelf2 100% OK  (target move)
  33 · bob            | shelf2 | shelf0  32% XX  |  shelf2 100% OK 
  34 · moves          | shelf2 | shelf2  30% OK  |  shelf2 100% OK 
  35 · book           | shelf2 | shelf2  42% OK  |  shelf2 100% OK 
  36 · down           | shelf2 | shelf3  32% XX  |  shelf2 100% OK 
  37 · one            | shelf2 | shelf3  39% XX  |  shelf2 100% OK 
  38 · shelves        | shelf2 | shelf3  44% XX  |  shelf2 100% OK 
  39 · .              | shelf2 | shelf2  63% OK  |  shelf2 100% OK 
  40 · alice          | shelf2 | shelf3  36% XX  |  shelf2 100% OK  (target move)
  41 · moves          | shelf2 | shelf3  36% XX  |  shelf2 100% OK  (target move)
  42 · ball           | shelf2 | shelf3  31% XX  |  shelf2 100% OK  (target move)
  43 · down           | shelf2 | shelf0  26% XX  |  shelf2  98% OK  (target move)
  44 · one            | shelf2 | shelf3  39% XX  |  shelf2  99% OK  (target move)
  45 · shelves        | shelf2 | shelf2  35% OK  |  shelf2  99% OK  (target move)
  46 · .              | shelf1 | shelf0  30% XX  |  shelf1 100% OK  (target move)
  47 · carol          | shelf1 | shelf2  29% XX  |  shelf1 100% OK 
  48 · moves          | shelf1 | shelf2  36% XX  |  shelf1 100% OK 
  49 · book           | shelf1 | shelf3  31% XX  |  shelf1 100% OK 
  50 · up             | shelf1 | shelf2  27% XX  |  shelf1  99% OK 
  51 · one            | shelf1 | shelf3  28% XX  |  shelf1  99% OK 
  52 · shelves        | shelf1 | shelf2  34% XX  |  shelf1  99% OK 
  53 · .              | shelf1 | shelf3  29% XX  |  shelf1 100% OK 
        QUERY → answer | shelf1 |    shelf1 100% |      shelf1 100%

monitor read-accuracy across the story:  GPT 56%   NextLat-1step 100%

## Example 3  (true final shelf = 3)

> the ball starts on shelf0 . the book starts on shelf3 . bob moves book up one shelves . bob moves ball up one shelves . bob moves ball up one shelves . carol moves book down one shelves . carol moves ball up one shelves . bob moves book down one shelves . where is the ball ? shelf3 end

           pos · token |   TRUE |      GPT probe |  NextLat-1 probe
----------------------------------------------------------------------
   0 · the            | shelf0 | shelf1  26% XX  |  shelf1  26% XX 
   1 · ball           | shelf0 | shelf1  26% XX  |  shelf1  26% XX 
   2 · starts         | shelf0 | shelf1  26% XX  |  shelf1  26% XX 
   3 · on             | shelf0 | shelf1  26% XX  |  shelf1  26% XX 
   4 · shelf0         | shelf0 | shelf0 100% OK  |  shelf0 100% OK 
   5 · .              | shelf0 | shelf0 100% OK  |  shelf0 100% OK 
   6 · the            | shelf0 | shelf0 100% OK  |  shelf0 100% OK 
   7 · book           | shelf0 | shelf0 100% OK  |  shelf0 100% OK 
   8 · starts         | shelf0 | shelf0 100% OK  |  shelf0 100% OK 
   9 · on             | shelf0 | shelf0 100% OK  |  shelf0 100% OK 
  10 · shelf3         | shelf0 | shelf0 100% OK  |  shelf0 100% OK 
  11 · .              | shelf0 | shelf0 100% OK  |  shelf0 100% OK 
  12 · bob            | shelf0 | shelf0 100% OK  |  shelf0 100% OK 
  13 · moves          | shelf0 | shelf0 100% OK  |  shelf0 100% OK 
  14 · book           | shelf0 | shelf0  99% OK  |  shelf0 100% OK 
  15 · up             | shelf0 | shelf0  92% OK  |  shelf0 100% OK 
  16 · one            | shelf0 | shelf0  99% OK  |  shelf0 100% OK 
  17 · shelves        | shelf0 | shelf0 100% OK  |  shelf0 100% OK 
  18 · .              | shelf0 | shelf0 100% OK  |  shelf0 100% OK 
  19 · bob            | shelf0 | shelf0  90% OK  |  shelf0 100% OK  (target move)
  20 · moves          | shelf0 | shelf0  99% OK  |  shelf0 100% OK  (target move)
  21 · ball           | shelf0 | shelf0  76% OK  |  shelf0 100% OK  (target move)
  22 · up             | shelf0 | shelf0  56% OK  |  shelf0 100% OK  (target move)
  23 · one            | shelf0 | shelf0  92% OK  |  shelf0 100% OK  (target move)
  24 · shelves        | shelf0 | shelf0  99% OK  |  shelf0 100% OK  (target move)
  25 · .              | shelf1 | shelf1  56% OK  |  shelf1 100% OK  (target move)
  26 · bob            | shelf1 | shelf1  40% OK  |  shelf1 100% OK  (target move)
  27 · moves          | shelf1 | shelf2  47% XX  |  shelf1 100% OK  (target move)
  28 · ball           | shelf1 | shelf1  40% OK  |  shelf1 100% OK  (target move)
  29 · up             | shelf1 | shelf3  31% XX  |  shelf1 100% OK  (target move)
  30 · one            | shelf1 | shelf1  49% OK  |  shelf1 100% OK  (target move)
  31 · shelves        | shelf1 | shelf1  55% OK  |  shelf1 100% OK  (target move)
  32 · .              | shelf2 | shelf1  40% XX  |  shelf2 100% OK  (target move)
  33 · carol          | shelf2 | shelf0  29% XX  |  shelf2 100% OK 
  34 · moves          | shelf2 | shelf3  36% XX  |  shelf2 100% OK 
  35 · book           | shelf2 | shelf0  50% XX  |  shelf2 100% OK 
  36 · down           | shelf2 | shelf0  29% XX  |  shelf2 100% OK 
  37 · one            | shelf2 | shelf1  36% XX  |  shelf2 100% OK 
  38 · shelves        | shelf2 | shelf0  39% XX  |  shelf2 100% OK 
  39 · .              | shelf2 | shelf1  47% XX  |  shelf2 100% OK 
  40 · carol          | shelf2 | shelf1  31% XX  |  shelf2 100% OK  (target move)
  41 · moves          | shelf2 | shelf0  40% XX  |  shelf2 100% OK  (target move)
  42 · ball           | shelf2 | shelf0  43% XX  |  shelf2 100% OK  (target move)
  43 · up             | shelf2 | shelf3  26% XX  |  shelf2  99% OK  (target move)
  44 · one            | shelf2 | shelf1  33% XX  |  shelf2  99% OK  (target move)
  45 · shelves        | shelf2 | shelf0  30% XX  |  shelf2  95% OK  (target move)
  46 · .              | shelf3 | shelf3  42% OK  |  shelf3 100% OK  (target move)
  47 · bob            | shelf3 | shelf1  33% XX  |  shelf3 100% OK 
  48 · moves          | shelf3 | shelf0  33% XX  |  shelf3 100% OK 
  49 · book           | shelf3 | shelf2  36% XX  |  shelf3 100% OK 
  50 · down           | shelf3 | shelf2  33% XX  |  shelf3  96% OK 
  51 · one            | shelf3 | shelf0  30% XX  |  shelf3  95% OK 
  52 · shelves        | shelf3 | shelf0  26% XX  |  shelf3  95% OK 
  53 · .              | shelf3 | shelf2  33% XX  |  shelf3 100% OK 
        QUERY → answer | shelf3 |    shelf3 100% |       shelf3 99%

monitor read-accuracy across the story:  GPT 50%   NextLat-1step 93%

## Example 4  (true final shelf = 2)

> the ball starts on shelf3 . the book starts on shelf0 . alice moves ball up one shelves . alice moves ball down one shelves . bob moves ball down one shelves . alice moves book up one shelves . alice moves book up one shelves . bob moves book down one shelves . where is the ball ? shelf2 end

           pos · token |   TRUE |      GPT probe |  NextLat-1 probe
----------------------------------------------------------------------
   0 · the            | shelf3 | shelf1  26% XX  |  shelf1  26% XX 
   1 · ball           | shelf3 | shelf1  26% XX  |  shelf1  26% XX 
   2 · starts         | shelf3 | shelf1  26% XX  |  shelf1  26% XX 
   3 · on             | shelf3 | shelf1  26% XX  |  shelf1  26% XX 
   4 · shelf3         | shelf3 | shelf3 100% OK  |  shelf3 100% OK 
   5 · .              | shelf3 | shelf3 100% OK  |  shelf3 100% OK 
   6 · the            | shelf3 | shelf3 100% OK  |  shelf3 100% OK 
   7 · book           | shelf3 | shelf3 100% OK  |  shelf3 100% OK 
   8 · starts         | shelf3 | shelf3 100% OK  |  shelf3 100% OK 
   9 · on             | shelf3 | shelf3 100% OK  |  shelf3 100% OK 
  10 · shelf0         | shelf3 | shelf3 100% OK  |  shelf3 100% OK 
  11 · .              | shelf3 | shelf3 100% OK  |  shelf3 100% OK 
  12 · alice          | shelf3 | shelf3 100% OK  |  shelf3 100% OK  (target move)
  13 · moves          | shelf3 | shelf3 100% OK  |  shelf3 100% OK  (target move)
  14 · ball           | shelf3 | shelf3  99% OK  |  shelf3 100% OK  (target move)
  15 · up             | shelf3 | shelf3  92% OK  |  shelf3 100% OK  (target move)
  16 · one            | shelf3 | shelf3 100% OK  |  shelf3 100% OK  (target move)
  17 · shelves        | shelf3 | shelf3 100% OK  |  shelf3 100% OK  (target move)
  18 · .              | shelf0 | shelf0 100% OK  |  shelf0 100% OK  (target move)
  19 · alice          | shelf0 | shelf0  77% OK  |  shelf0 100% OK  (target move)
  20 · moves          | shelf0 | shelf0 100% OK  |  shelf0 100% OK  (target move)
  21 · ball           | shelf0 | shelf0  80% OK  |  shelf0 100% OK  (target move)
  22 · down           | shelf0 | shelf3  51% XX  |  shelf0  99% OK  (target move)
  23 · one            | shelf0 | shelf0  55% OK  |  shelf0 100% OK  (target move)
  24 · shelves        | shelf0 | shelf0  66% OK  |  shelf0 100% OK  (target move)
  25 · .              | shelf3 | shelf3  44% OK  |  shelf3 100% OK  (target move)
  26 · bob            | shelf3 | shelf1  37% XX  |  shelf3 100% OK  (target move)
  27 · moves          | shelf3 | shelf3  69% OK  |  shelf3 100% OK  (target move)
  28 · ball           | shelf3 | shelf2  46% XX  |  shelf3 100% OK  (target move)
  29 · down           | shelf3 | shelf3  36% OK  |  shelf3 100% OK  (target move)
  30 · one            | shelf3 | shelf1  44% XX  |  shelf3 100% OK  (target move)
  31 · shelves        | shelf3 | shelf1  68% XX  |  shelf3 100% OK  (target move)
  32 · .              | shelf2 | shelf3  38% XX  |  shelf2 100% OK  (target move)
  33 · alice          | shelf2 | shelf2  38% OK  |  shelf2 100% OK 
  34 · moves          | shelf2 | shelf2  28% OK  |  shelf2 100% OK 
  35 · book           | shelf2 | shelf2  33% OK  |  shelf2 100% OK 
  36 · up             | shelf2 | shelf3  30% XX  |  shelf2 100% OK 
  37 · one            | shelf2 | shelf2  37% OK  |  shelf2 100% OK 
  38 · shelves        | shelf2 | shelf2  45% OK  |  shelf2 100% OK 
  39 · .              | shelf2 | shelf3  34% XX  |  shelf2 100% OK 
  40 · alice          | shelf2 | shelf3  32% XX  |  shelf2 100% OK 
  41 · moves          | shelf2 | shelf1  34% XX  |  shelf2 100% OK 
  42 · book           | shelf2 | shelf0  30% XX  |  shelf2 100% OK 
  43 · up             | shelf2 | shelf3  28% XX  |  shelf2 100% OK 
  44 · one            | shelf2 | shelf3  40% XX  |  shelf2 100% OK 
  45 · shelves        | shelf2 | shelf2  35% OK  |  shelf2 100% OK 
  46 · .              | shelf2 | shelf2  28% OK  |  shelf2 100% OK 
  47 · bob            | shelf2 | shelf2  33% OK  |  shelf2 100% OK 
  48 · moves          | shelf2 | shelf1  37% XX  |  shelf2 100% OK 
  49 · book           | shelf2 | shelf1  33% XX  |  shelf2 100% OK 
  50 · down           | shelf2 | shelf0  26% XX  |  shelf2  98% OK 
  51 · one            | shelf2 | shelf1  28% XX  |  shelf2  98% OK 
  52 · shelves        | shelf2 | shelf1  30% XX  |  shelf2  98% OK 
  53 · .              | shelf2 | shelf3  29% XX  |  shelf2 100% OK 
        QUERY → answer | shelf2 |    shelf2 100% |      shelf2 100%

monitor read-accuracy across the story:  GPT 57%   NextLat-1step 93%