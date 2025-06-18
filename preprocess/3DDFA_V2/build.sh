cd FaceBoxes
sh ./build_cpu_nms.sh
cd ..

cd Sim3DR
sh ./build_sim3dr.sh
cd ..

cd utils_3ddfa/asset
gcc -shared -Wall -O3 render.c -o render.so -fPIC
cd ../..
