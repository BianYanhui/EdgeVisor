#ifndef NN_KV_REUSED_ALGORITHM_HPP
#define NN_KV_REUSED_ALGORITHM_HPP

#include <vector>
#include <cstdint>

struct Range {
    int start;
    int end;
    
    int count() const { return end - start; }
};

struct DeviceStatus {
    int device_id;
    double execution_time_ms;
    
    // Active computing ranges
    Range current_head_compute;
    Range current_ffn_compute;
    
    // KV Cache holding ranges
    Range kv_head_holding;
    // FFN usually doesn't have KV cache constraints, but keeping structure consistent
    Range kv_ffn_holding; 
};

struct RebalanceMove {
    int from_device_id;
    int to_device_id;
    int cmdKind; // 1 for move
    int headMove;
    int ffnMove;
};

// Function prototype
std::vector<RebalanceMove> RebalanceHeadMoves(const std::vector<DeviceStatus>& devices);

#endif // NN_KV_REUSED_ALGORITHM_HPP