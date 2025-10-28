// Example 8: Classes Inside Namespaces
// Shows how to organize classes using namespaces

#include <iostream>
#include <string>

// Namespace containing related classes
namespace Banking {
    class Account {
    private:
        std::string owner;
        double balance;

    public:
        Account(std::string name, double initial) : owner(name), balance(initial) {}

        void deposit(double amount) {
            balance += amount;
            std::cout << "Deposited: $" << amount << std::endl;
        }

        void withdraw(double amount) {
            if (balance >= amount) {
                balance -= amount;
                std::cout << "Withdrawn: $" << amount << std::endl;
            }
        }

        void display() {
            std::cout << "Account owner: " << owner << ", Balance: $" << balance << std::endl;
        }
    };

    class Transaction {
    public:
        static void transfer(Account& from, Account& to, double amount) {
            std::cout << "Transferring $" << amount << std::endl;
            from.withdraw(amount);
            to.deposit(amount);
        }
    };
}

// Another namespace with different classes
namespace Shopping {
    class Cart {
    private:
        double total;

    public:
        Cart() : total(0) {}

        void addItem(double price) {
            total += price;
            std::cout << "Added item: $" << price << std::endl;
        }

        double getTotal() { return total; }
    };
}



int main() {
    // Create objects from Banking namespace
    Banking::Account account1("Alice", 1000.0);
    Banking::Account account2("Bob", 500.0);

    account1.display();
    account2.display();

    Banking::Transaction::transfer(account1, account2, 200.0);

    account1.display();
    account2.display();

    // Create object from Shopping namespace
    Shopping::Cart myCart;
    myCart.addItem(29.99);
    myCart.addItem(49.99);
    std::cout << "Cart total: $" << myCart.getTotal() << std::endl;

    return 0;
}
