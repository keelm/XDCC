# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2018.3.1\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2018.3.1\bin\cmake\win\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\Simon\Documents\vm_share\xgboost_further_optimized

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles\runxgboost.dir\depend.make

# Include the progress variables for this target.
include CMakeFiles\runxgboost.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\runxgboost.dir\flags.make

CMakeFiles\runxgboost.dir\src\cli_main.cc.obj: CMakeFiles\runxgboost.dir\flags.make
CMakeFiles\runxgboost.dir\src\cli_main.cc.obj: ..\src\cli_main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/runxgboost.dir/src/cli_main.cc.obj"
	C:\PROGRA~2\MICROS~1\2017\BUILDT~1\VC\Tools\MSVC\1416~1.270\bin\Hostx86\x86\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\runxgboost.dir\src\cli_main.cc.obj /FdCMakeFiles\runxgboost.dir\ /FS -c C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\src\cli_main.cc
<<

CMakeFiles\runxgboost.dir\src\cli_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runxgboost.dir/src/cli_main.cc.i"
	C:\PROGRA~2\MICROS~1\2017\BUILDT~1\VC\Tools\MSVC\1416~1.270\bin\Hostx86\x86\cl.exe > CMakeFiles\runxgboost.dir\src\cli_main.cc.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\src\cli_main.cc
<<

CMakeFiles\runxgboost.dir\src\cli_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runxgboost.dir/src/cli_main.cc.s"
	C:\PROGRA~2\MICROS~1\2017\BUILDT~1\VC\Tools\MSVC\1416~1.270\bin\Hostx86\x86\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\runxgboost.dir\src\cli_main.cc.s /c C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\src\cli_main.cc
<<

# Object files for target runxgboost
runxgboost_OBJECTS = \
"CMakeFiles\runxgboost.dir\src\cli_main.cc.obj"

# External object files for target runxgboost
runxgboost_EXTERNAL_OBJECTS = \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\c_api\c_api.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\c_api\c_api_error.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\common\common.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\common\hist_util.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\common\host_device_vector.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\data\data.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\data\simple_csr_source.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\data\simple_dmatrix.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\data\sparse_page_dmatrix.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\data\sparse_page_raw_format.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\data\sparse_page_source.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\data\sparse_page_writer.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\gbm\gblinear.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\gbm\gbm.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\gbm\gbtree.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\learner.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\linear\linear_updater.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\linear\updater_coordinate.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\linear\updater_shotgun.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\logging.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\metric\elementwise_metric.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\metric\metric.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\metric\multiclass_metric.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\metric\multilabel_metric.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\metric\rank_metric.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\objective\multiclass_obj.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\objective\multilabel_obj.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\objective\objective.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\objective\rank_obj.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\objective\regression_obj.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\predictor\cpu_predictor.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\predictor\predictor.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\tree\tree_model.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\tree\tree_updater.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\tree\updater_fast_hist.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\tree\updater_histmaker.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\tree\updater_multilabel.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\tree\updater_prune.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\tree\updater_refresh.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\tree\updater_skmaker.cc.obj" \
"C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\objxgboost.dir\src\tree\updater_sync.cc.obj"

..\xgboost.exe: CMakeFiles\runxgboost.dir\src\cli_main.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\c_api\c_api.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\c_api\c_api_error.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\common\common.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\common\hist_util.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\common\host_device_vector.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\data\data.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\data\simple_csr_source.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\data\simple_dmatrix.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\data\sparse_page_dmatrix.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\data\sparse_page_raw_format.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\data\sparse_page_source.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\data\sparse_page_writer.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\gbm\gblinear.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\gbm\gbm.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\gbm\gbtree.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\learner.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\linear\linear_updater.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\linear\updater_coordinate.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\linear\updater_shotgun.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\logging.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\metric\elementwise_metric.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\metric\metric.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\metric\multiclass_metric.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\metric\multilabel_metric.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\metric\rank_metric.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\objective\multiclass_obj.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\objective\multilabel_obj.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\objective\objective.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\objective\rank_obj.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\objective\regression_obj.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\predictor\cpu_predictor.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\predictor\predictor.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\tree\tree_model.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\tree\tree_updater.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\tree\updater_fast_hist.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\tree\updater_histmaker.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\tree\updater_multilabel.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\tree\updater_prune.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\tree\updater_refresh.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\tree\updater_skmaker.cc.obj
..\xgboost.exe: CMakeFiles\objxgboost.dir\src\tree\updater_sync.cc.obj
..\xgboost.exe: CMakeFiles\runxgboost.dir\build.make
..\xgboost.exe: dmlc-core\dmlccore.lib
..\xgboost.exe: rabit.lib
..\xgboost.exe: CMakeFiles\runxgboost.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ..\xgboost.exe"
	"C:\Program Files\JetBrains\CLion 2018.3.1\bin\cmake\win\bin\cmake.exe" -E vs_link_exe --intdir=CMakeFiles\runxgboost.dir --manifests  -- C:\PROGRA~2\MICROS~1\2017\BUILDT~1\VC\Tools\MSVC\1416~1.270\bin\Hostx86\x86\link.exe /nologo @CMakeFiles\runxgboost.dir\objects1.rsp @<<
 /out:..\xgboost.exe /implib:xgboost.lib /pdb:C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\xgboost.pdb /version:0.0  /machine:X86 /debug /INCREMENTAL /subsystem:console dmlc-core\dmlccore.lib rabit.lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib 
<<

# Rule to build all files generated by this target.
CMakeFiles\runxgboost.dir\build: ..\xgboost.exe

.PHONY : CMakeFiles\runxgboost.dir\build

CMakeFiles\runxgboost.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\runxgboost.dir\cmake_clean.cmake
.PHONY : CMakeFiles\runxgboost.dir\clean

CMakeFiles\runxgboost.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\Simon\Documents\vm_share\xgboost_further_optimized C:\Users\Simon\Documents\vm_share\xgboost_further_optimized C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug C:\Users\Simon\Documents\vm_share\xgboost_further_optimized\cmake-build-debug\CMakeFiles\runxgboost.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\runxgboost.dir\depend
