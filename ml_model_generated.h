#pragma once

// ============================================
// Auto-generated from Decision Tree (max_depth=4)
// ============================================

struct MLPrediction {
    bool should_split;
    double confidence;
};

MLPrediction predict_split_ml(double variance, double grad_h, double grad_v) {
    if (variance <= 5141.600098) {
        if (grad_h <= 62.218750) {
            if (grad_h <= 18.218750) {
                if (grad_h <= 2.312500) {
                    // Samples: no_split=1, split=0
                    return {false, 1.0000};
                } else {
                    // Samples: no_split=0, split=0
                    return {false, 0.5909};
                }
            } else {
                // Samples: no_split=1, split=0
                return {false, 1.0000};
            }
        } else {
            // Samples: no_split=0, split=1
            return {true, 1.0000};
        }
    } else {
        if (variance <= 5754.145020) {
            if (variance <= 5577.659912) {
                if (variance <= 5193.639893) {
                    // Samples: no_split=0, split=1
                    return {true, 1.0000};
                } else {
                    // Samples: no_split=0, split=0
                    return {false, 0.7037};
                }
            } else {
                if (grad_h <= 38.132799) {
                    // Samples: no_split=0, split=1
                    return {true, 1.0000};
                } else {
                    // Samples: no_split=0, split=0
                    return {true, 0.7500};
                }
            }
        } else {
            if (variance <= 8911.485352) {
                if (grad_v <= 13.085940) {
                    // Samples: no_split=0, split=0
                    return {false, 0.6000};
                } else {
                    // Samples: no_split=0, split=0
                    return {false, 0.9714};
                }
            } else {
                // Samples: no_split=0, split=1
                return {true, 1.0000};
            }
        }
    }
}