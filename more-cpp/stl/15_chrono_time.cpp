/**
 * 15_chrono_time.cpp
 *
 * CHRONO AND TIME UTILITIES
 * - Duration types
 * - Time points
 * - Clocks
 * - Time measurement
 * - Date/time operations (C++20)
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== CHRONO AND TIME ===\n";

    using namespace std::chrono;

    separator("DURATIONS");

    // 1. Duration Types
    std::cout << "\n1. DURATION TYPES:\n";
    hours h(2);
    minutes m(30);
    seconds s(45);
    milliseconds ms(500);
    microseconds us(1000);
    nanoseconds ns(1000000);

    std::cout << "2 hours = " << h.count() << " hours\n";
    std::cout << "30 minutes = " << m.count() << " minutes\n";
    std::cout << "45 seconds = " << s.count() << " seconds\n";
    std::cout << "500 milliseconds = " << ms.count() << " ms\n";

    // 2. Duration Arithmetic
    std::cout << "\n2. DURATION ARITHMETIC:\n";
    auto total = h + m + s;
    std::cout << "Total: " << duration_cast<seconds>(total).count() << " seconds\n";
    std::cout << "Total: " << duration_cast<minutes>(total).count() << " minutes\n";

    auto diff = hours(5) - minutes(30);
    std::cout << "5 hours - 30 minutes = " << duration_cast<minutes>(diff).count() << " minutes\n";

    // 3. Duration Conversion
    std::cout << "\n3. DURATION CONVERSION:\n";
    seconds sec(120);
    auto as_minutes = duration_cast<minutes>(sec);
    auto as_ms = duration_cast<milliseconds>(sec);

    std::cout << sec.count() << " seconds\n";
    std::cout << "  = " << as_minutes.count() << " minutes\n";
    std::cout << "  = " << as_ms.count() << " milliseconds\n";

    // 4. Literal Suffixes (C++14)
    std::cout << "\n4. LITERAL SUFFIXES:\n";
    using namespace std::chrono_literals;

    auto one_hour = 1h;
    auto thirty_min = 30min;
    auto ten_sec = 10s;
    auto fifty_ms = 50ms;
    auto hundred_us = 100us;
    auto thousand_ns = 1000ns;

    std::cout << "1h = " << duration_cast<minutes>(one_hour).count() << " minutes\n";
    std::cout << "30min = " << thirty_min.count() << " minutes\n";
    std::cout << "10s + 50ms = " << (ten_sec + fifty_ms).count() << " milliseconds\n";

    separator("CLOCKS");

    // 5. System Clock
    std::cout << "\n5. SYSTEM_CLOCK:\n";
    auto sys_now = system_clock::now();
    std::time_t now_time = system_clock::to_time_t(sys_now);
    std::cout << "System time: " << std::ctime(&now_time);

    // 6. Steady Clock
    std::cout << "\n6. STEADY_CLOCK (monotonic):\n";
    auto steady_start = steady_clock::now();
    std::this_thread::sleep_for(100ms);
    auto steady_end = steady_clock::now();

    auto elapsed = duration_cast<milliseconds>(steady_end - steady_start);
    std::cout << "Elapsed time: " << elapsed.count() << " ms\n";

    // 7. High Resolution Clock
    std::cout << "\n7. HIGH_RESOLUTION_CLOCK:\n";
    auto hr_start = high_resolution_clock::now();
    // Some computation
    long long sum = 0;
    for (int i = 0; i < 1000000; ++i) {
        sum += i;
    }
    auto hr_end = high_resolution_clock::now();

    auto duration_us = duration_cast<microseconds>(hr_end - hr_start);
    std::cout << "Computation took: " << duration_us.count() << " microseconds\n";

    separator("TIME POINTS");

    // 8. Time Points
    std::cout << "\n8. TIME POINTS:\n";
    time_point<system_clock> tp_now = system_clock::now();
    time_point<system_clock> tp_future = tp_now + hours(24);

    std::time_t now_tt = system_clock::to_time_t(tp_now);
    std::time_t future_tt = system_clock::to_time_t(tp_future);

    std::cout << "Now: " << std::ctime(&now_tt);
    std::cout << "24 hours later: " << std::ctime(&future_tt);

    // 9. Time Point Arithmetic
    std::cout << "\n9. TIME POINT ARITHMETIC:\n";
    auto start_tp = system_clock::now();
    auto end_tp = start_tp + seconds(30);
    auto diff_duration = end_tp - start_tp;

    std::cout << "Difference: " << duration_cast<seconds>(diff_duration).count() << " seconds\n";

    separator("TIME MEASUREMENT");

    // 10. Measure Function Execution
    std::cout << "\n10. MEASURE FUNCTION EXECUTION:\n";

    auto measure = [](auto&& func, const std::string& name) {
        auto start = high_resolution_clock::now();
        func();
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        std::cout << name << " took: " << duration.count() << " us\n";
    };

    measure([]() {
        std::this_thread::sleep_for(10ms);
    }, "Sleep 10ms");

    measure([]() {
        int sum = 0;
        for (int i = 0; i < 100000; ++i) sum += i;
    }, "Sum loop");

    // 11. Timer Class
    std::cout << "\n11. TIMER CLASS:\n";

    class Timer {
        time_point<high_resolution_clock> start_time;
    public:
        Timer() : start_time(high_resolution_clock::now()) {}

        void reset() {
            start_time = high_resolution_clock::now();
        }

        double elapsed_seconds() const {
            auto end_time = high_resolution_clock::now();
            return duration_cast<duration<double>>(end_time - start_time).count();
        }

        long long elapsed_ms() const {
            auto end_time = high_resolution_clock::now();
            return duration_cast<milliseconds>(end_time - start_time).count();
        }
    };

    Timer timer;
    std::this_thread::sleep_for(150ms);
    std::cout << "Timer elapsed: " << timer.elapsed_ms() << " ms\n";
    std::cout << "Timer elapsed: " << timer.elapsed_seconds() << " seconds\n";

    separator("DELAYS AND TIMEOUTS");

    // 12. Sleep Functions
    std::cout << "\n12. SLEEP FUNCTIONS:\n";
    std::cout << "Sleeping for 100ms...\n";
    std::this_thread::sleep_for(100ms);
    std::cout << "Awake!\n";

    auto wake_time = system_clock::now() + seconds(1);
    std::cout << "Sleeping until specific time...\n";
    std::this_thread::sleep_until(wake_time);
    std::cout << "Awake at target time!\n";

    separator("PRACTICAL EXAMPLES");

    // 13. Benchmark Comparisons
    std::cout << "\n13. BENCHMARK COMPARISONS:\n";

    auto benchmark = [](auto&& func, int iterations) {
        auto start = high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        auto end = high_resolution_clock::now();
        return duration_cast<microseconds>(end - start).count();
    };

    auto method1_time = benchmark([]() { int x = 5 * 5; }, 1000000);
    auto method2_time = benchmark([]() { int x = 5; x *= 5; }, 1000000);

    std::cout << "Method 1 time: " << method1_time << " us\n";
    std::cout << "Method 2 time: " << method2_time << " us\n";

    // 14. Timeout Check
    std::cout << "\n14. TIMEOUT CHECK:\n";

    auto timeout_start = steady_clock::now();
    auto timeout_duration = seconds(2);

    while (true) {
        auto current = steady_clock::now();
        if (current - timeout_start >= timeout_duration) {
            std::cout << "Timeout reached!\n";
            break;
        }
        std::this_thread::sleep_for(500ms);
        std::cout << "Working...\n";
    }

    // 15. Rate Limiting
    std::cout << "\n15. RATE LIMITING:\n";

    auto last_action = steady_clock::now();
    auto min_interval = milliseconds(100);

    for (int i = 0; i < 5; ++i) {
        auto now = steady_clock::now();
        auto elapsed = now - last_action;

        if (elapsed < min_interval) {
            std::this_thread::sleep_for(min_interval - elapsed);
        }

        std::cout << "Action " << i + 1 << " executed\n";
        last_action = steady_clock::now();
    }

    // 16. Format Duration
    std::cout << "\n16. FORMAT DURATION:\n";

    auto format_duration = [](milliseconds ms) {
        auto h = duration_cast<hours>(ms);
        ms -= duration_cast<milliseconds>(h);
        auto m = duration_cast<minutes>(ms);
        ms -= duration_cast<milliseconds>(m);
        auto s = duration_cast<seconds>(ms);
        ms -= duration_cast<milliseconds>(s);

        std::cout << h.count() << "h " << m.count() << "m "
                  << s.count() << "s " << ms.count() << "ms\n";
    };

    milliseconds total_ms(3725500);  // 1h 2m 5s 500ms
    std::cout << "Duration: ";
    format_duration(total_ms);

    std::cout << "\n=== END OF CHRONO AND TIME ===\n";

    return 0;
}
