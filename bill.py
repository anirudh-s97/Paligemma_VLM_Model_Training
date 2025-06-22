import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Tuple
import json

class BillSplitter:
    def __init__(self, root):
        self.root = root
        self.root.title("Roommate Bill Splitter")
        self.root.geometry("1000x700")
        
        # Data storage
        self.roommates = []
        self.products = []
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        self.setup_ui(main_frame)
        
    def setup_ui(self, parent):
        # Title
        title_label = ttk.Label(parent, text="Walmart Bill Splitter", font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Roommates setup section
        roommates_frame = ttk.LabelFrame(parent, text="Setup Roommates", padding="10")
        roommates_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        roommates_frame.columnconfigure(1, weight=1)
        
        ttk.Label(roommates_frame, text="Enter roommate names (up to 5):").grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        self.roommate_entries = []
        for i in range(5):
            ttk.Label(roommates_frame, text=f"Person {i+1}:").grid(row=i+1, column=0, sticky=tk.W, padx=(0, 10))
            entry = ttk.Entry(roommates_frame, width=20)
            entry.grid(row=i+1, column=1, sticky=(tk.W, tk.E), pady=2)
            self.roommate_entries.append(entry)
        
        update_btn = ttk.Button(roommates_frame, text="Update Roommates", command=self.update_roommates)
        update_btn.grid(row=6, column=0, columnspan=2, pady=10)
        
        # Products entry section
        products_frame = ttk.LabelFrame(parent, text="Add Products", padding="10")
        products_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        products_frame.columnconfigure(1, weight=1)
        products_frame.columnconfigure(3, weight=1)
        
        ttk.Label(products_frame, text="Product Name:").grid(row=0, column=0, sticky=tk.W)
        self.product_name_entry = ttk.Entry(products_frame, width=30)
        self.product_name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 20))
        
        ttk.Label(products_frame, text="Price ($):").grid(row=0, column=2, sticky=tk.W)
        self.product_price_entry = ttk.Entry(products_frame, width=15)
        self.product_price_entry.grid(row=0, column=3, sticky=(tk.W, tk.E), padx=(5, 0))
        
        ttk.Label(products_frame, text="Assign to:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        
        # Assignment options
        self.assignment_var = tk.StringVar(value="common")
        assignment_frame = ttk.Frame(products_frame)
        assignment_frame.grid(row=1, column=1, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Common option
        ttk.Radiobutton(assignment_frame, text="Common (split equally)", 
                       variable=self.assignment_var, value="common").grid(row=0, column=0, sticky=tk.W)
        
        # Individual assignment
        ttk.Radiobutton(assignment_frame, text="Specific person(s):", 
                       variable=self.assignment_var, value="specific").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # Checkboxes for individual assignment
        self.person_vars = []
        self.person_checkboxes = []
        checkbox_frame = ttk.Frame(assignment_frame)
        checkbox_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(5, 0))
        
        for i in range(5):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(checkbox_frame, text=f"Person {i+1}", variable=var, state="disabled")
            cb.grid(row=i//3, column=i%3, sticky=tk.W, padx=(20, 20))
            self.person_vars.append(var)
            self.person_checkboxes.append(cb)
        
        # Bind radio button to enable/disable checkboxes
        self.assignment_var.trace('w', self.on_assignment_change)
        
        add_product_btn = ttk.Button(products_frame, text="Add Product", command=self.add_product)
        add_product_btn.grid(row=3, column=0, columnspan=4, pady=20)
        
        # Products list
        list_frame = ttk.LabelFrame(parent, text="Products Added", padding="10")
        list_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(1, weight=1)
        
        # Treeview for products
        columns = ('Product', 'Price', 'Assignment')
        self.products_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.products_tree.heading(col, text=col)
            self.products_tree.column(col, width=100)
        
        self.products_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.products_tree.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.products_tree.configure(yscrollcommand=scrollbar.set)
        
        # Remove button
        remove_btn = ttk.Button(list_frame, text="Remove Selected", command=self.remove_product)
        remove_btn.grid(row=2, column=0, pady=10)
        
        # Results section
        results_frame = ttk.LabelFrame(parent, text="Bill Summary", padding="10")
        results_frame.grid(row=3, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        self.results_text = tk.Text(results_frame, width=40, height=15, wrap=tk.WORD)
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        results_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        # Calculate button
        calc_btn = ttk.Button(results_frame, text="Calculate Bills", command=self.calculate_bills)
        calc_btn.grid(row=2, column=0, pady=10)
        
        # Clear all button
        clear_btn = ttk.Button(results_frame, text="Clear All", command=self.clear_all)
        clear_btn.grid(row=3, column=0, pady=5)
        
    def on_assignment_change(self, *args):
        """Enable/disable checkboxes based on assignment type"""
        if self.assignment_var.get() == "specific":
            for cb in self.person_checkboxes:
                cb.configure(state="normal")
        else:
            for cb in self.person_checkboxes:
                cb.configure(state="disabled")
            # Clear all checkboxes
            for var in self.person_vars:
                var.set(False)
    
    def update_roommates(self):
        """Update roommate list from entries"""
        self.roommates = []
        for i, entry in enumerate(self.roommate_entries):
            name = entry.get().strip()
            if name:
                self.roommates.append(name)
                # Update checkbox text
                self.person_checkboxes[i].configure(text=name)
            else:
                self.person_checkboxes[i].configure(text=f"Person {i+1}")
        
        if not self.roommates:
            messagebox.showwarning("Warning", "Please enter at least one roommate name.")
            return
        
        messagebox.showinfo("Success", f"Updated with {len(self.roommates)} roommates: {', '.join(self.roommates)}")
        self.calculate_bills()  # Recalculate with new roommates
    
    def add_product(self):
        """Add a product to the list"""
        if not self.roommates:
            messagebox.showwarning("Warning", "Please set up roommates first.")
            return
        
        name = self.product_name_entry.get().strip()
        price_str = self.product_price_entry.get().strip()
        
        if not name or not price_str:
            messagebox.showwarning("Warning", "Please enter both product name and price.")
            return
        
        try:
            price = float(price_str)
            if price < 0:
                raise ValueError("Price cannot be negative")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid price.")
            return
        
        # Determine assignment
        if self.assignment_var.get() == "common":
            assignment = "Common"
            assigned_to = list(range(len(self.roommates)))  # All roommates
        else:
            assigned_indices = [i for i, var in enumerate(self.person_vars) if var.get() and i < len(self.roommates)]
            if not assigned_indices:
                messagebox.showwarning("Warning", "Please select at least one person for specific assignment.")
                return
            
            assigned_names = [self.roommates[i] for i in assigned_indices]
            assignment = ", ".join(assigned_names)
            assigned_to = assigned_indices
        
        # Add product
        product = {
            'name': name,
            'price': price,
            'assignment': assignment,
            'assigned_to': assigned_to
        }
        self.products.append(product)
        
        # Add to treeview
        self.products_tree.insert('', 'end', values=(name, f"${price:.2f}", assignment))
        
        # Clear entries
        self.product_name_entry.delete(0, tk.END)
        self.product_price_entry.delete(0, tk.END)
        for var in self.person_vars:
            var.set(False)
        self.assignment_var.set("common")
        
        # Recalculate bills
        self.calculate_bills()
    
    def remove_product(self):
        """Remove selected product from the list"""
        selected = self.products_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a product to remove.")
            return
        
        # Get index of selected item
        index = self.products_tree.index(selected[0])
        
        # Remove from data and treeview
        del self.products[index]
        self.products_tree.delete(selected[0])
        
        # Recalculate bills
        self.calculate_bills()
    
    def calculate_bills(self):
        """Calculate and display bills for each roommate"""
        if not self.roommates:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Please set up roommates first.")
            return
        
        if not self.products:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No products added yet.")
            return
        
        # Initialize bills
        bills = {name: 0.0 for name in self.roommates}
        
        # Calculate bills
        for product in self.products:
            price_per_person = product['price'] / len(product['assigned_to'])
            for person_index in product['assigned_to']:
                if person_index < len(self.roommates):
                    bills[self.roommates[person_index]] += price_per_person
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        
        total = sum(product['price'] for product in self.products)
        self.results_text.insert(tk.END, f"WALMART BILL SUMMARY\n")
        self.results_text.insert(tk.END, "=" * 30 + "\n\n")
        
        self.results_text.insert(tk.END, f"Total Bill Amount: ${total:.2f}\n\n")
        
        self.results_text.insert(tk.END, "INDIVIDUAL BILLS:\n")
        self.results_text.insert(tk.END, "-" * 20 + "\n")
        
        for name, amount in bills.items():
            self.results_text.insert(tk.END, f"{name}: ${amount:.2f}\n")
        
        self.results_text.insert(tk.END, f"\nVerification: ${sum(bills.values()):.2f} (should equal total)\n\n")
        
        # Product breakdown
        self.results_text.insert(tk.END, "PRODUCT BREAKDOWN:\n")
        self.results_text.insert(tk.END, "-" * 20 + "\n")
        
        for product in self.products:
            self.results_text.insert(tk.END, f"• {product['name']}: ${product['price']:.2f}\n")
            self.results_text.insert(tk.END, f"  Assigned to: {product['assignment']}\n")
            if len(product['assigned_to']) > 1:
                cost_per_person = product['price'] / len(product['assigned_to'])
                self.results_text.insert(tk.END, f"  Cost per person: ${cost_per_person:.2f}\n")
            self.results_text.insert(tk.END, "\n")
    
    def clear_all(self):
        """Clear all data"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all data?"):
            self.products = []
            self.products_tree.delete(*self.products_tree.get_children())
            self.results_text.delete(1.0, tk.END)
            
            # Clear entries
            self.product_name_entry.delete(0, tk.END)
            self.product_price_entry.delete(0, tk.END)
            for var in self.person_vars:
                var.set(False)
            self.assignment_var.set("common")

def main():
    root = tk.Tk()
    app = BillSplitter(root)
    root.mainloop()

if __name__ == "__main__":
    main()