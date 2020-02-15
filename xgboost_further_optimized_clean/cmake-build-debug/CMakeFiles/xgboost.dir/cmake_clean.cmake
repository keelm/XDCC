file(REMOVE_RECURSE
  "../lib/xgboost.pdb"
  "../lib/xgboost.dll"
  "xgboost.lib"
  "../lib/xgboost.dll.manifest"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/xgboost.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
