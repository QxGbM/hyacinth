
#include <nccl.h>
#include <cstdint>
#include <vector>
#include <fstream>
#include <string>

int32_t main(int32_t argc, char* argv[]) {
  std::string file = 1 < argc ? std::string(argv[1]) : std::string("id.txt");

  std::ofstream stream(file);
  if (stream.is_open()) {
    std::vector<uint8_t> arr(sizeof(ncclUniqueId));
    ncclUniqueId id; ncclGetUniqueId(&id);
    std::memcpy(arr.data(), &id, sizeof(ncclUniqueId));
    for (uint8_t byte : arr) stream << int32_t(byte) << " ";
    stream << std::endl;
    stream.close();
  }

  return 0;
}
