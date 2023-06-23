file(REMOVE_RECURSE
  "GPP.pdb"
  "GPP.so"
  "__init__.py"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/GPP.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
