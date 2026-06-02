
"""
You can change the text color; its an odd string, though. To start to print in a different color, you MUST use the string `\033[1;XXm`, where `XX` is one of the following:  
* 30 (Black)  
* 31 (Red)  
* 32 (Green)  
* 33 (Yellow)  
* 34 (Blue)  
* 35 (Magenta)  
* 36 (Cyan)  
* 37 (White)  

And then, when you are done with the color, you MUST print `\033[0m`.  

Fortunately, you can store the odd start and end sequences in a variable, and then just straight reference them in a formatted string and it will work.

Examples:

**without tags
print(f"\n\033[1;31mThis is red\033[0m \033[1;32mAnd this is green.\033[0m")

**With used tags
print(f"{green_start}This starts at green{end} and then reverts to white.")

"""
class ColoredText:
    BLACK_TEXT = "\033[1;30m"
    RED_TEXT = "\033[1;31m"
    GREEN_TEXT = "\033[1;32m"
    YELLOW_TEXT = "\033[1;33m"
    BLUE_TEXT = "\033[1;34m"
    MAGENTA_TEXT = "\033[1;35m"
    CYAN_TEXT = "\033[1;36m"
    WHITE_TEXT = "\033[1;37m"
    END_TEXT = "\033[0m" 


