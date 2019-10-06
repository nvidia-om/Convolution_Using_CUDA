# Convolution Using CUDA
A project to compare time taken by CPU and GPU to carry out convolution operation

# CUDA Installation
* Installer: CUDA Toolkit 10.1 Update 2 from https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=7&target_type=exelocal
   * Install CUDA and follow on screen prompt
   * Once installed:
      * To verify correct configuration of hardware and software, build and run the deviceQuery sample program.
         * Solution path: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\1_Utilities\deviceQuery\deviceQuery_vs2015.sln
      * To ensure that the system and the CUDA-capable device are able to communicate correctly, build and run the bandwidthTest program.
         * Solution path: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\1_Utilities\bandwidthTest\bandwidthTest_vs2015.sln
      * To see a graphical representation of what CUDA can do, run the sample Particles program	
         * Solution path: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\5_Simulations\particles\particles_vs2015.sln
      * Output exe will be in path: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\bin\win64\Release
    
# Project Setup in Visual Studio 2015 Professional
-> File->New->Project
-> Select Templates->NVIDIA->CUDA 10.1->CUDA 10.1 Runtime
-> Give project a name. E.g. 2D_Convolution_Using_Shared_Memory
-> Go to "Properties" of the project:
	-> Set "Output Directory" and "Intermediate Directory" under "General" tab as:
		Output Directory : $(SolutionDir)output\$(Configuration)\
		Intermediate Directory : $(SolutionDir)build\$(ProjectName)_$(Configuration)\
	-> Set "Code Generation" under "CUDA C/C++->Device" tab as:
		Code Generation : compute_30,sm_30;compute_35,sm_35;compute_37,sm_37;compute_50,sm_50;compute_52,sm_52;compute_60,sm_60;compute_61,sm_61;compute_70,sm_70;compute_75,sm_75;
