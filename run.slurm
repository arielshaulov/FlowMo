#! /bin/sh

#SBATCH --job-name=Wan_freeinit
#SBATCH --output=/home/ai_center/ai_users/itaytuviah/video-motion/jobs-out-err/%x-%j.out
#SBATCH --error=/home/ai_center/ai_users/itaytuviah/video-motion/jobs-out-err/%x-%j.err
#SBATCH --open-mode=append      # Append instead of overwriting logs
#SBATCH --time=1500 # max time (minutes)
#SBATCH --gpus=1 # GPUs in total
#SBATCH --mem-per-gpu=55000
#SBATCH --partition=gpu-ai
#SBATCH --constraint=h100  # Enforce using an H100 GPU

srun --gres=gpu:1 nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export PYTORCH_KERNEL_CACHE_PATH="/home/ai_center/ai_users/itaytuviah/video-motion/cache"
export TORCH_KERNEL_CACHE_DIR="/home/ai_center/ai_users/itaytuviah/video-motion/cache"
export HF_HOME="/home/ai_center/ai_users/itaytuviah/video-motion/cache"
export TRANSFORMERS_CACHE="/home/ai_center/ai_users/itaytuviah/video-motion/cache"
export WANDB_CACHE_DIR="/home/ai_center/ai_users/itaytuviah/video-motion/cache"
export WANDB_DIR="/home/ai_center/ai_users/itaytuviah/video-motion/cache/wandb/"
export WANDB_CONFIG_DIR="/home/ai_center/ai_users/itaytuviah/video-motion/cache/wandb/"
export WANDB_TEMP_DIR="/home/ai_center/ai_users/itaytuviah/video-motion/cache/wandb/"
export WANDB_DATA_DIR="/home/ai_center/ai_users/itaytuviah/video-motion/cache/wandb/"


python generate.py --task t2v-1.3B \
                    --size 832*480 \
                    --ckpt_dir ./Wan2.1-T2V-1.3B \
                    --prompts "A woman performing an intricate dance on stage, illuminated by a single spotlight in the first frame."   \
                    --seeds 1024 

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A man juggling with three red balls in a city street." "A white kitten playing with a ball." "A slow-motion shot captures a runner’s legs as they dash through a busy intersection." "A young girl playing basketball in a red brick wall background." "A basketball game in progress, with two players reaching up to grab the ball." \
#                     --seeds 31 \
#                     --optimize "True" \

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A group of basketballs floating in mid-air in slow motion." "A dog playing with an orange ball with blue stripes." "A woman doing acrobatic exercises on a pole in the gym." "A young man performing a cartwheel on a gray surface." "A golden retriever playing fetch on a grassy field." \
#                     --seeds 31 \
#                     --optimize "True" \


##################################### USER STUDY ###################################


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A woman doing a headstand on a beach." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A woman engaging in a challenging workout routine, performing pull-ups on green bars." "Two ibexes navigating a rocky hillside."  \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A close-up of a runner’s legs as they sprint through a crowded city street." "Athletic man doing gymnastics elements on horizontal bar in city park." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A man performing a handstand on a wooden deck overlooking a green lake." "Young adult female performs an air gymnastic show on circus arena." "A woman enjoying the fun of hula hooping." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \



# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A small dog playing with a red ball on a hardwood floor." "A woman engaging in a lively trampoline workout. The woman jumps and exercises on the trampoline. The background is a room with white walls and a white ceiling, and there are two large windows on the left side of the wall, and a mirror on the right side reflecting the womans image."  \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \



# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A man juggling with three red balls in a city street." "A white kitten playing with a ball." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A slow-motion shot captures a runner’s legs as they dash through a busy intersection, dodging cars and pedestrians, the city life bustling around them." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A young girl playing basketball in a red brick wall background. The girl, with fair skin and long blonde hair, is wearing a green jacket and has her left arm up to throw the ball. In the mid-frame, the girl is still playing basketball, with her right hand holding the ball in front of her face. The ground is dark gray cement with some patches of grass growing through it. As the video progresses, the girl is seen playing near some grassy areas on the ground." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A basketball game in progress, with two players reaching up to grab the ball as it spills out of the net. The player on the left has his hand outstretched, while the player on the right has both hands raised high. The ball is just above their
#                     fingertips, indicating that they are both trying to grab it simultaneously. The background of the image is blurred, but it
#                     appears to be a gymnasium or sports arena, with fluorescent lights illuminating the scene. As the video progresses, the
#                     players continue to jump and stretch to gain possession of the ball, their movements becoming more urgent and intense.
#                     The ball flies back and forth between them, with neither player able to secure it. In the final frame, the ball is still in
#                     mid-air, the players hands reaching up to grab it as the video ends" \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A group of basketballs floating in mid-air in slow motion, with a larger ball on the left and two smaller balls on either side in the initial frame. Overall, the video captures the dynamic and energetic movement of basketballs as they float and bounce through space." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A dog playing with an orange ball with blue stripes. The dog picks up the ball and holds it in its mouth, conveying a sense of playfulness and energy. Throughout the video, the dog is seen playing with the ball, capturing the joy and excitement of the moment." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A woman doing acrobatic exercises on a pole in the gym." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A young man performing a cartwheel on a gray surface. He is dressed in orange pants, a black t-shirt, and white sneakers. As he executes the cartwheel, his right arm is extended upward, and his left arm is bent at the elbow, reaching down to the ground. His right leg is extended behind him, while his left leg is bent at the knee, pointing towards the camera. The background is a featureless gray wall. The mans energy and focus are evident as he completes the cartwheel, showcasing his athleticism and coordination." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \



# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A golden retriever playing fetch on a grassy field. The dog is running with a frisbee in its mouth, its fur waving in the wind" "A brightly colored ball spins rapidly on a flat surface, its patterns blurring as it twirls in place."\
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A basketball spins on a player's fingertip, maintaining balance while gradually slowing down." "A person jogs along a forest trail at dawn, their feet kicking up dirt with every stride, the sunlight filtering through the trees casting long shadows on the path."\
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A child jumps up and down in place, their feet leaving the ground briefly before landing again." "A person lifts one knee high in a marching motion, then places their foot back down and repeats with the other leg." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "Professional cyclist training indoors on a stationary bike trainer." "Young Adult Male Doing Handstand on the beach." "A young woman practicing boxing in a gym." "A man jumping in a pool." "A man doing push-ups on a ledge overlooking a body of water. The man appears to be doing a push-up, with his head down." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A man enjoying a leisurely bike ride along a road next to a body of water during a sunset. As he pedals, he looks down at his front wheel, seemingly focused on his ride. The background features a large body of water, with a gray wall along the left side of the road in the mid-frame caption" "close up shot of the feet of a woman exercising on a cardio fitness machine in a fitness club. As the video progresses, the legs continue to pedal the bike in a smooth, consistent motion."\
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A woman engaging in an intense workout on a stationary bike while monitoring her progress on a screen." "A woman running along a river with a city skyline in the background." "A skier walking up a snowy hill with their skis on their back and ski poles in hand." "A woman running through a grassy area, wearing a black tank top, gray and white leggings, and white sneakers. She
#                     is initially running on a dirt path, surrounded by trees with green leaves. As she continues to run, the scenery changes
#                     to a park, and her leggings change to a blue and white pattern. She is still running on a dirt path, surrounded by trees
#                     and green grass. The video captures her journey as she runs through the grassy area, enjoying the outdoors and the
#                     beauty of nature" "A young girl coloring at her desk." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A close-up of a runner’s legs as they dash through a rainstorm, their shoes splashing through puddles as they push
#                     forward with determination." "Tracking camera shot. A kangaroo hops swiftly across an open grassy plain." "A close-up view of a spiral object with a glowing center. The object appears to be made of metal and has a shiny, reflective surface. This light creates a series of concentric circles around the objects circumference, which are visible
#                     due to the reflection of the light off the metal surface." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A roulette wheel in a dimly lit room or casino floor. In the center of the wheel, there’s a small white ball that appears to be spinning rapidly as it moves around the track. The ball spins around the wheel, and the wheel rotates counterclockwise." "A close-up of a jogger’s feet as they run along a rocky coastal path, their shoes gripping the uneven surface, with the ocean waves crashing below." "A person’s hands as they shape and mold clay on a pottery wheel. The hands are covered in brown clay and are visible from the elbows down, with the forearms resting on top of a large yellow pottery wheel."\
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A conveyor belt pouring out a large amount of small, brown objects into a pile on the ground. The objects being poured are falling from the conveyor belt in a steady stream, forming a large pile on the ground below. In the background, the sky is bright blue and cloudless, providing a stark contrast to the darker colors of the conveyor belt and the pile of objects." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A 3d rendering of coins and small objects floating against a black background. The coins are gold, silver, bronze, and copper, with various denominations and sizes. Some have a shiny finish, while others are matte or tarnished. The
#                     scene is chaotic and dynamic, with the objects seemingly flying around in all directions. As the video progresses, the
#                     coins and objects tumble and spin, creating a sense of movement and energy. By the end, the screen is filled with white
#                     objects of various shapes and sizes, suggesting that something exciting is happening." "A puppy runs through a grassy field." "A cinematic shot of a person walking along a quiet country road, their feet crunching on the gravel with every step, fields of wheat swaying in the breeze on either side." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \
                    

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A washing machine undergoing a full cycle. It begins with a top-down view of the machine filled with water and white soap suds, with two black rubber seals on either side of the stainless steel drum. The video progresses to show the drum spinning, with the suds becoming more agitated and the seals moving along with the drums motion." "Sweet Cherries on Stems Colliding and Splashing Water Droplets" "A series of colorful balloons floating in mid-air, creating a festive and celebratory atmosphere." "A cinematic shot of a person jogging along a riverside path, their feet rhythmically tapping against the ground, the river flowing gently beside them." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \
                    

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A green helicopter taking off from an airport runway." "A hand holding a yellow fidget spinner. The hand is fair-skinned and holds the bright yellow fidget spinner with silver bearings. The background is blurred and appears to be trees against a blue sky. The video captures the subtle movements of the hand as it spins the fidget spinner, creating a soothing and mesmerizing visual effect. As the video progresses, the hand continues to hold the fidget spinner, showcasing its smooth and satisfying motion. The background remains blurred, adding a sense of tranquility to the scene. Overall, the video is a calming and enjoyable display of the simple pleasure of fidget spinning." "A windmill spinning in a green field." "A bicycle wheel spins forward, moving in a circular motion while keeping balance." "A waterwheel turns as water flows over it, the paddles rotating consistently." "A close-up of a person’s legs as they walk through a sun-dappled forest, the light playing off their shoes as they navigate the uneven terrain." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A man riding a mountain bike on a dirt trail." "A child's toy top spins on a smooth surface, rotating without stopping." "A basketball spins on a player's fingertip, showcasing balance and skill." "A jellyfish swimming in shallow water. The jellyfish has a translucent body with a distinctive pattern of white circles and lines. It appears to be swimming just below the surface of the water, which is dark and murky due to the presence of algae or other aquatic plants." "A cinematic shot of a person walking along a cobblestone street in a historic town, their feet making a rhythmic tap on the stones as they move." "A group of horses grazing in a grassy field behind a black wooden fence" "A fish swims forward in a steady line, its tail swaying side to side as it propels itself." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A penguin waddles in a straight line, shifting from one foot to the other." "A man is jumping rope on the sandy beach, with waves crashing in the background." "A man enjoying water skiing on a brown river with a green shore and lily pads in the background. Water sprays up from underneath him as he skis across the surface of the lake." "A man is swimming in a clear blue pool, enjoying the cool water and the freedom of movement in the pool. As he continues to swim, he glides gracefully through the water, his arms and legs moving in a smooth and coordinated rhythm." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A kid running in the mountains of Campo Imperatore, Italy, at the sunset. He is wearing a red polo shirt, blue jeans, and brown shoes. As he runs, he passes by some white rocks on the ground." "A woman doing push-up exercise on a beach at sunset."\
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A woman is shown running through a field, with tall grass and wildflowers all around her. She is a fair-skinned woman with long, red hair, wearing a black t-shirt and leggings, and listening to music on her phone. In the background, there are trees and more fields of greenery." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A man exercising with battle ropes at a gym." "A person engaging in a boxing workout at a gym."  \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A dark gray horse running in an enclosed corral. It is running towards the camera." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A close-up of a runner’s legs as they dash up a flight of stairs in a city park, their feet hitting each step with precision and power." "A man is swimming in the ocean. In the background, the sky is hazy and overexposed, with the sun shining brightly above the horizon. As the video progresses, the man continues to swim, his arms moving rhythmically through the water." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A herd of white cows walking down a dirt path. The cows are all facing forward and walking towards the right side of the image. The background is blurry but appears to be a field or pasture." "A person jogs along a trail in a dense forest, their legs pumping as they navigate the roots and rocks that dot the path." "A young woman dances in the night bustle against the backdrop of a glowing fanfare." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A man is walking down the street while pushing a trash can. The man, wearing a red t-shirt, blue jeans, and brown sandals, pushes the black trash can on wheels." "A man enjoying a mountain biking adventure through a forest. He is seen riding a black and white mountain bike down a dirt path, with his back to the camera." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \                  

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "Women's legs walk into the sea with waves." "A young man walking on a treadmill. He is wearing a white tank top and red shorts, and has his hands on the sides of the machine as he runs." "Closeup of feet of a professional soccer player training with ball on stadium field with artificial turf."\
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \   

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A helicopter flying over a forest. The helicopter is black and has two large rotor blades on top. It is flying low to the ground, with its nose pointing slightly upwards." "A close-up of a person’s feet as they walk through a field of wildflowers, their shoes brushing against the blooms with each step." "A man is playing basketball, dribbling the ball and making shots." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \



# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A giraffe running through an open field. The background is a bright blue sky with fluffy white clouds." "A person jogs along a city waterfront, their legs moving steadily as the sun sets, casting a warm glow over the water and the buildings behind them." "A woman is doing push-ups on a mat in the studio." "Two dancers perform on a stage. The man stands behind the woman with his left arm is lifted over his head and the other is stretched to the right. The woman lets go of the man’s right hand, swinging her leg to the left and performing a pirouette. She spins four times and ends up facing the man."  \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \



# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A woman drinks from a water bottle in a forest. The woman has fair skin and brown hair. She is wearing a black jacket and black and white gloves." "Tracking camera shot. A polar bear walks across a snowy landscape. It looks curiously around as it plods through the snow. The background is a snowy landscape with footprints visible in the snow. Sunlight shines from overhead and casts the bear’s shadow on the snow." "A cinematic shot of a person walking through a desert at midday, their legs moving slowly but steadily across the sand dunes, with heat waves distorting the distant horizon."  \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A man jumping rope on a dark stage. His movements are fluid and energetic. Two spotlights shine down from above him." "A woman twirls a hula hoop around her waist in a park during sunset. The woman, with medium-length curly black hair and a yellow tank top, stands on a grassy field surrounded by trees. As the hoop revolves around her waist, she shifts her hips rhythmically to keep it moving. The golden sunlight casts a long shadow behind her." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A man exercises on a leg press machine at a gym." "A young woman enjoys a cup of coffee on a balcony." "A man energetically bangs on a drum kit. He holds drumsticks in both hands and bashes on the drum kit with the drumsticks." "A woman performs high knees on a beach." "Aerial tracking camera shot. A white semi-truck drives on a highway." "A woman is holding a clear wine glass partly filled with a burgundy-colored wine. Facing forward, the woman smiles, she raises the glass with her left hand and takes a small sip"  \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A man works on a piece of wood in a workroom. He holds a shiny silver chisel with a wooden handle in his right hand." "Sliced green apples are tossed in a brown liquid. The apples are cut into thick slices and have shiny green skins with some light-colored speckling. They begin to rotate clockwise, flying out in every direction as the light amber liquid splashes and swirls behind them." "A baboon eats a mango." "A young woman vapes in the living room. The woman exhales the thick, billowing smoke." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \



# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts  "A woman performing an aerial hoop trick. The woman hangs from a black aerial hoop attached to the ceiling by a rope. In the initial frame, she has her legs wrapped around the hoop and her arms extended outward, holding onto the hoop with both hands. Her body is twisted, looking up towards the ceiling, with her shadow cast on the white wall behind her. As the video progresses, she continues to hang from the hoop, her body twisted in various positions, her arms and legs wrapped around the hoop as she performs the aerial trick. The background remains the same, with shadows from the aerial hoop and the woman’s body on the white wall."\
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \
                    

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "Modern urban street ballet dancer performing acrobatics and jumps." "A woman doing a pirouette in an empty dance studio." "A woman dancing hip hop, street dancing in the studio. Slow motion." "A brunette woman doing some acrobatic elements on aerial hoop outdoors." "A woman, with long brown hair and wearing a black top and gray bottoms, climbs on a pole with her right leg wrapped around it and her left arm extended upward. The background is a white wall with a mirror reflecting the woman’s images."\
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A man performing a backflip. Slow motion." "A woman dancing in a gym. The woman is spinning around repeatedly." "A group of duck are walking in a row, one after the other. The background is a Japanese temple." "Arc camera shot. A young woman doing stretches on a beach." "A woman walking through a field of beautiful sunflowers. She spins counterclockwise and laughs. A field of shoulderlength sunflowers grow in the background, with trees on the horizon stretching up towards a cloudy sky" "Arc camera shot. A man playing the guitar." "A boy blowing out candles on a birthday cake." "A cheetah running in the Savannah." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "Tracking shot. A golden retriever runs through a grassy park. The dog’s ears flop up and down with each bounding step, and its tongue hangs out to one side. A frisbee flies into view from the left, and the dog leaps into the air to catch it. A group of people in the background claps and cheers." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A young girl skips down a quiet suburban street lined with trees. She has light brown skin and long, wavy black hair tied back with a red ribbon. The girl wears a white t-shirt, a denim skirt, and bright yellow sneakers. Her arms swing loosely as she skips." "A woman doing sit-ups at a gym." "A child riding his bicycle on a dirt path. The background is a dirt path lined with trees on either side." "A runner moves at full speed along a suburban sidewalk. The background is rows of houses and trees passing by in a blur." \
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A young woman engaging in a boxing workout. She is wearing red boxing gloves and a white t-shirt, and has long blonde hair. In the first frame, she is standing in front of a black punching bag, with her right arm extended and her left arm bent, ready to punch the bag. She appears focused and determined. In the second frame, she has moved to the left of the bag and is looking towards the right side of the image. She continues to punch the bag with her right arm extended and her left arm bent. In the final frame, she is still standing to the left of the bag and is looking towards the right side of the image. She is still wearing her red boxing gloves and white t-shirt, and her long blonde hair is visible. The background of a blue wall with a window on the left and a doorway on the right, as well as two black objects hanging from the ceiling. Throughout the video, the woman is intensely focused on her workout, punching the bag with precision and skill." "A brown bear walks in a grassy field."\
#                     --seeds 1024 \
#                     --optimize "True" \
#                     --lr 0.001 \

                    

# ##################################### USER STUDY ###################################



# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "Athletic man doing gymnastics elements on horizontal bar in city park. Male sportsmen perform strength exercises outdoors."  \
# #                   --seeds 11 12 13  \
# #                   --optimize "True" \

# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "A man jumping rope on a dark stage. His movements are fluid and energetic. Two spotlights shine down from above him." \
# #                   --seeds 11 12 13  \
# #                   --optimize "True" \

# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "A young woman enjoys a cup of coffee on a balcony."  "A woman, with long brown hair and wearing a black top and gray bottoms, climbs on a pole with her right leg wrapped around it and her left arm extended upward. The background is a white wall with a mirror reflecting the woman's images." \
# #                   --seeds 11 12 13  \
# #                   --optimize "True" \

# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "A woman, with long brown hair and wearing a black top and gray bottoms, climbs on a pole with her right leg wrapped around it and her left arm extended upward. The background is a white wall with a mirror reflecting the woman's images." \
# #                   --seeds 11 12 13  \
# #                   --optimize "True" \



# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "A man performing a backflip. Slow motion." \
# #                   --seeds 11 12 13  \
# #                   --optimize "True" \

# ################################################################################



# # # ################### HARD
# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "A man performing a backflip. Slow motion." \
# #                   --seeds 11  \
# #                   --optimize "True" \


# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "A woman, with long brown hair and wearing a black top and gray bottoms, climbs on a pole with her right leg wrapped around it and her left arm extended upward. The background is a white wall with a mirror reflecting the woman's images." \
# #                   --seeds 11 \
# #                   --optimize "True" \


# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "A man jumping rope on a dark stage. His movements are fluid and energetic. Two spotlights shine down from above him." \
# #                   --seeds 11  \
# #                   --optimize "True" \


# # ################### EASY

# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "A young woman enjoys a cup of coffee on a balcony." \
# #                   --seeds 11  \
# #                   --optimize "True" \


# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "A golden retriever playing fetch on a grassy field." \
# #                   --seeds 11 \
# #                   --optimize "True" \

# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "A penguin waddles in a straight line, shifting from one foot to the other." \
# #                   --seeds 11 \
# #                   --optimize "True" \

# # ################### MID

# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "A group of basketballs floating in mid-air in slow motion." \
# #                   --seeds 11 \
# #                   --optimize "True" \


# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "Tracking camera shot. A polar bear walks across a snowy landscape. It looks curiously around as it plods through the snow. The background is a snowy landscape with footprints visible in the snow. Sunlight shines from overhead and casts the bear’s shadow on the snow." \
# #                   --seeds 11 \
# #                   --optimize "True" \


# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "A woman doing a pirouette in an empty dance studio." \
# #                   --seeds 11 \
# #                   --optimize "True" \


# # python generate.py --task t2v-1.3B \
# #                    --size 832*480 \
# #                   --ckpt_dir ./Wan2.1-T2V-1.3B \
# #                   --prompts "A baboon eats a mango." \
# #                   --seeds 11 \
# #                   --optimize "True" \
