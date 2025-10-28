/**
 * 10_real_world_example.cpp
 *
 * TOPIC: Real-World Exception Handling Example
 *
 * This file demonstrates:
 * - A complete real-world application with proper exception handling
 * - Banking system with accounts, transactions, and error handling
 * - Combining all concepts: custom exceptions, RAII, exception safety
 * - Logging and error recovery
 * - Best practices in action
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>

using namespace std;

// ============================================================================
// CUSTOM EXCEPTIONS
// ============================================================================

class BankingException : public exception {
protected:
    string message;
public:
    BankingException(const string& msg) : message(msg) {}
    const char* what() const noexcept override { return message.c_str(); }
};

class InsufficientFundsException : public BankingException {
private:
    double requested;
    double available;
public:
    InsufficientFundsException(double req, double avail)
        : BankingException("Insufficient funds"), requested(req), available(avail) {
        ostringstream oss;
        oss << "Insufficient funds: requested $" << requested
            << ", available $" << available;
        message = oss.str();
    }
    double getRequested() const { return requested; }
    double getAvailable() const { return available; }
};

class InvalidAccountException : public BankingException {
public:
    InvalidAccountException(const string& accountId)
        : BankingException("Invalid account: " + accountId) {}
};

class TransactionLimitException : public BankingException {
public:
    TransactionLimitException(double amount, double limit)
        : BankingException("Transaction limit exceeded") {
        ostringstream oss;
        oss << "Transaction amount $" << amount
            << " exceeds limit $" << limit;
        message = oss.str();
    }
};

// ============================================================================
// LOGGER (RAII for file handling)
// ============================================================================

class Logger {
private:
    static Logger* instance;
    ofstream logFile;

    Logger() {
        logFile.open("bank_transactions.log", ios::app);
        if (!logFile.is_open()) {
            throw runtime_error("Cannot open log file");
        }
    }

public:
    static Logger& getInstance() {
        if (!instance) {
            instance = new Logger();
        }
        return *instance;
    }

    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }

    void log(const string& level, const string& message) noexcept {
        try {
            time_t now = time(nullptr);
            char timestamp[26];
            ctime_r(&now, timestamp);
            timestamp[24] = '\0';  // Remove newline

            logFile << "[" << timestamp << "] "
                   << "[" << level << "] "
                   << message << endl;
            logFile.flush();

            // Also print to console
            cout << "   [LOG:" << level << "] " << message << endl;
        }
        catch (...) {
            // Logger should never throw
            cerr << "   [ERROR] Logger failed!" << endl;
        }
    }

    void info(const string& msg) noexcept { log("INFO", msg); }
    void error(const string& msg) noexcept { log("ERROR", msg); }
    void warning(const string& msg) noexcept { log("WARNING", msg); }
};

Logger* Logger::instance = nullptr;

// ============================================================================
// TRANSACTION (Value object with exception safety)
// ============================================================================

class Transaction {
public:
    enum class Type { DEPOSIT, WITHDRAWAL, TRANSFER };

private:
    Type type;
    double amount;
    string fromAccount;
    string toAccount;
    bool successful;
    string timestamp;

    string getCurrentTimestamp() {
        time_t now = time(nullptr);
        char buffer[26];
        ctime_r(&now, buffer);
        buffer[24] = '\0';
        return string(buffer);
    }

public:
    Transaction(Type t, double amt, const string& from, const string& to = "")
        : type(t), amount(amt), fromAccount(from), toAccount(to),
          successful(false), timestamp(getCurrentTimestamp()) {}

    void markSuccessful() noexcept { successful = true; }
    bool isSuccessful() const noexcept { return successful; }
    double getAmount() const noexcept { return amount; }

    string toString() const {
        ostringstream oss;
        oss << timestamp << " - ";
        switch (type) {
            case Type::DEPOSIT:
                oss << "DEPOSIT $" << amount << " to " << fromAccount;
                break;
            case Type::WITHDRAWAL:
                oss << "WITHDRAWAL $" << amount << " from " << fromAccount;
                break;
            case Type::TRANSFER:
                oss << "TRANSFER $" << amount << " from " << fromAccount
                    << " to " << toAccount;
                break;
        }
        oss << " [" << (successful ? "SUCCESS" : "FAILED") << "]";
        return oss.str();
    }
};

// ============================================================================
// ACCOUNT (Exception-safe operations)
// ============================================================================

class Account {
private:
    string accountId;
    string ownerName;
    double balance;
    double dailyWithdrawalLimit;
    double dailyWithdrawalTotal;
    vector<shared_ptr<Transaction>> transactions;

    void validateAmount(double amount) const {
        if (amount <= 0) {
            throw invalid_argument("Amount must be positive");
        }
    }

public:
    Account(const string& id, const string& owner, double initialBalance)
        : accountId(id), ownerName(owner), balance(initialBalance),
          dailyWithdrawalLimit(1000.0), dailyWithdrawalTotal(0.0) {

        if (id.empty() || owner.empty()) {
            throw invalid_argument("Account ID and owner name required");
        }
        if (initialBalance < 0) {
            throw invalid_argument("Initial balance cannot be negative");
        }

        Logger::getInstance().info("Account created: " + id + " for " + owner);
    }

    string getId() const noexcept { return accountId; }
    string getOwner() const noexcept { return ownerName; }
    double getBalance() const noexcept { return balance; }

    // Strong exception guarantee
    void deposit(double amount) {
        validateAmount(amount);

        auto transaction = make_shared<Transaction>(
            Transaction::Type::DEPOSIT, amount, accountId);

        try {
            balance += amount;
            transaction->markSuccessful();
            transactions.push_back(transaction);

            Logger::getInstance().info(transaction->toString());
        }
        catch (...) {
            Logger::getInstance().error("Deposit failed: " + accountId);
            throw;
        }
    }

    // Strong exception guarantee
    void withdraw(double amount) {
        validateAmount(amount);

        auto transaction = make_shared<Transaction>(
            Transaction::Type::WITHDRAWAL, amount, accountId);

        // Check balance
        if (balance < amount) {
            Logger::getInstance().error(transaction->toString());
            throw InsufficientFundsException(amount, balance);
        }

        // Check daily limit
        if (dailyWithdrawalTotal + amount > dailyWithdrawalLimit) {
            Logger::getInstance().error(transaction->toString());
            throw TransactionLimitException(amount, dailyWithdrawalLimit - dailyWithdrawalTotal);
        }

        try {
            balance -= amount;
            dailyWithdrawalTotal += amount;
            transaction->markSuccessful();
            transactions.push_back(transaction);

            Logger::getInstance().info(transaction->toString());
        }
        catch (...) {
            Logger::getInstance().error("Withdrawal failed: " + accountId);
            throw;
        }
    }

    void resetDailyLimit() noexcept {
        dailyWithdrawalTotal = 0.0;
        Logger::getInstance().info("Daily limit reset for account: " + accountId);
    }

    void displayStatement() const noexcept {
        cout << "\n   ===== Account Statement =====" << endl;
        cout << "   Account: " << accountId << endl;
        cout << "   Owner: " << ownerName << endl;
        cout << "   Balance: $" << fixed << setprecision(2) << balance << endl;
        cout << "   Daily Withdrawal Used: $" << dailyWithdrawalTotal
             << " / $" << dailyWithdrawalLimit << endl;
        cout << "   Recent Transactions:" << endl;

        size_t start = transactions.size() > 5 ? transactions.size() - 5 : 0;
        for (size_t i = start; i < transactions.size(); i++) {
            cout << "     " << transactions[i]->toString() << endl;
        }
        cout << "   =============================" << endl;
    }
};

// ============================================================================
// BANK (Manages accounts and transactions)
// ============================================================================

class Bank {
private:
    string bankName;
    vector<shared_ptr<Account>> accounts;

    shared_ptr<Account> findAccount(const string& accountId) {
        for (auto& account : accounts) {
            if (account->getId() == accountId) {
                return account;
            }
        }
        return nullptr;
    }

public:
    Bank(const string& name) : bankName(name) {
        Logger::getInstance().info("Bank '" + name + "' initialized");
    }

    void createAccount(const string& id, const string& owner, double initialBalance) {
        try {
            // Check if account already exists
            if (findAccount(id) != nullptr) {
                throw InvalidAccountException("Account " + id + " already exists");
            }

            auto account = make_shared<Account>(id, owner, initialBalance);
            accounts.push_back(account);

            cout << "   Account created successfully!" << endl;
        }
        catch (const BankingException& e) {
            Logger::getInstance().error(string("Create account failed: ") + e.what());
            throw;
        }
        catch (const exception& e) {
            Logger::getInstance().error(string("Unexpected error: ") + e.what());
            throw runtime_error("Account creation failed");
        }
    }

    void performDeposit(const string& accountId, double amount) {
        try {
            auto account = findAccount(accountId);
            if (!account) {
                throw InvalidAccountException(accountId);
            }

            account->deposit(amount);
            cout << "   Deposit successful!" << endl;
        }
        catch (const BankingException& e) {
            Logger::getInstance().error(string("Deposit failed: ") + e.what());
            throw;
        }
    }

    void performWithdrawal(const string& accountId, double amount) {
        try {
            auto account = findAccount(accountId);
            if (!account) {
                throw InvalidAccountException(accountId);
            }

            account->withdraw(amount);
            cout << "   Withdrawal successful!" << endl;
        }
        catch (const BankingException& e) {
            Logger::getInstance().error(string("Withdrawal failed: ") + e.what());
            throw;
        }
    }

    void performTransfer(const string& fromId, const string& toId, double amount) {
        // Strong exception guarantee using two-phase approach
        auto fromAccount = findAccount(fromId);
        auto toAccount = findAccount(toId);

        if (!fromAccount) throw InvalidAccountException(fromId);
        if (!toAccount) throw InvalidAccountException(toId);

        Logger::getInstance().info("Transfer: $" + to_string(amount) +
                                  " from " + fromId + " to " + toId);

        try {
            // Phase 1: Withdraw (can fail)
            fromAccount->withdraw(amount);

            try {
                // Phase 2: Deposit (should not fail if amount is valid)
                toAccount->deposit(amount);
                cout << "   Transfer successful!" << endl;
            }
            catch (...) {
                // Rollback: deposit back to source
                Logger::getInstance().error("Transfer failed - rolling back");
                fromAccount->deposit(amount);
                throw;
            }
        }
        catch (const BankingException& e) {
            Logger::getInstance().error(string("Transfer failed: ") + e.what());
            throw;
        }
    }

    void showAccount(const string& accountId) const {
        for (const auto& account : accounts) {
            if (account->getId() == accountId) {
                account->displayStatement();
                return;
            }
        }
        cout << "   Account not found: " << accountId << endl;
    }
};

// ============================================================================
// MAIN - Demonstrating real-world usage
// ============================================================================

int main() {
    cout << "=== Real-World Banking System Example ===" << endl;

    try {
        // Initialize bank
        Bank bank("FirstBank");

        // Create accounts
        cout << "\n1. Creating Accounts:" << endl;
        bank.createAccount("ACC001", "Alice Johnson", 5000.0);
        bank.createAccount("ACC002", "Bob Smith", 3000.0);
        bank.createAccount("ACC003", "Charlie Brown", 1000.0);

        // Successful operations
        cout << "\n2. Successful Operations:" << endl;
        bank.performDeposit("ACC001", 500.0);
        bank.performWithdrawal("ACC002", 200.0);
        bank.performTransfer("ACC001", "ACC002", 1000.0);

        // Display statements
        cout << "\n3. Account Statements:" << endl;
        bank.showAccount("ACC001");
        bank.showAccount("ACC002");

        // Error scenarios
        cout << "\n4. Error Scenarios:" << endl;

        // Invalid account
        cout << "\n   a) Invalid account:" << endl;
        try {
            bank.performDeposit("ACC999", 100.0);
        }
        catch (const InvalidAccountException& e) {
            cout << "   Caught: " << e.what() << endl;
        }

        // Insufficient funds
        cout << "\n   b) Insufficient funds:" << endl;
        try {
            bank.performWithdrawal("ACC003", 2000.0);
        }
        catch (const InsufficientFundsException& e) {
            cout << "   Caught: " << e.what() << endl;
            cout << "   Available: $" << e.getAvailable() << endl;
        }

        // Transaction limit
        cout << "\n   c) Daily limit exceeded:" << endl;
        try {
            bank.performWithdrawal("ACC001", 1200.0);
        }
        catch (const TransactionLimitException& e) {
            cout << "   Caught: " << e.what() << endl;
        }

        // Invalid amount
        cout << "\n   d) Invalid amount:" << endl;
        try {
            bank.performDeposit("ACC001", -100.0);
        }
        catch (const invalid_argument& e) {
            cout << "   Caught: " << e.what() << endl;
        }

        // Failed transfer (rollback)
        cout << "\n   e) Transfer with insufficient funds (rollback test):" << endl;
        cout << "   Before failed transfer:" << endl;
        bank.showAccount("ACC003");

        try {
            bank.performTransfer("ACC003", "ACC001", 2000.0);
        }
        catch (const InsufficientFundsException& e) {
            cout << "   Caught: " << e.what() << endl;
        }

        cout << "   After failed transfer (should be unchanged):" << endl;
        bank.showAccount("ACC003");

        cout << "\n=== Banking system completed successfully ===" << endl;
        cout << "\nCheck 'bank_transactions.log' for complete transaction history" << endl;

    }
    catch (const exception& e) {
        cerr << "\n[FATAL ERROR] Unhandled exception: " << e.what() << endl;
        return 1;
    }
    catch (...) {
        cerr << "\n[FATAL ERROR] Unknown exception occurred!" << endl;
        return 1;
    }

    /**
     * KEY CONCEPTS DEMONSTRATED:
     *
     * 1. Custom Exception Hierarchy
     *    - BankingException base class
     *    - Specific exceptions for different errors
     *    - Additional data in exceptions
     *
     * 2. RAII
     *    - Logger manages file handle
     *    - Automatic resource cleanup
     *    - No manual file closing needed
     *
     * 3. Exception Safety
     *    - Strong guarantee in transfers (rollback)
     *    - No resource leaks
     *    - Consistent state after errors
     *
     * 4. Smart Pointers
     *    - shared_ptr for accounts and transactions
     *    - Automatic memory management
     *    - No manual delete needed
     *
     * 5. noexcept
     *    - Used on getters and utility functions
     *    - Logger operations don't throw
     *    - Proper use of noexcept guarantee
     *
     * 6. Error Handling
     *    - Logging all operations
     *    - Graceful error recovery
     *    - User-friendly error messages
     *
     * 7. Best Practices
     *    - Validate input early
     *    - Use specific exception types
     *    - Log all errors
     *    - Maintain consistency
     *    - Clean up resources automatically
     */

    return 0;
}
