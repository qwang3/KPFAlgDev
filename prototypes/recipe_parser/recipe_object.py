import operator
import sys, os
import _ctypes

def di(obj_id):
    """ Inverse of id() function. """
    return _ctypes.PyObj_FromPtr(obj_id)

class Event:

    def __init__(self):
        
        # state what type of event this is
        self.type = None
        self.ref = None
    
    def update(self, other):
        # This is used to reference a actual event
        # (Operation or Condition). This is used so that 
        # we retain the object_id of this class when we add 
        # a actual event to the list (kinda important in linked list) 
        self.ref = other


## Operation class
# Operations() is a Event() that is meant to execute a 
# specific operaion on a input argument. 
# Each Operation() class, when constructed correctly, should
# have an associated input Argument() class and a 
# function handle.
# When executed successfully, it will point to the next 
# Event() class, which can be another Operation() or a 
# Condition()

class Operation(Event): 

    def __init__(self, name, call):
        '''
        constructor
        '''
        Event.__init__(self)
        # Specify that this is a operation 
        self.type = 'op'
        # Name of the operation
        self.name = name
        # Function handle to the operation
        self.call = call
        # Next event. Could be a operation or a condition 
        # This could also be None to signify EOF
        self.next = Event()

## Condition Class
# Condition() class is suppose to check a input Argument's field, and
# determine which Event() is happening next. 
# One Condition() class can point to many different Event() classes, 
# and each  Event() has an associated condition statement
class Condition(Event):

    def __init__(self):
        '''
        Condition constructor
        '''
        Event.__init__(self)
        # Specify that this is a condition 
        self.type = 'condi'
        # Dictionaries to store condition, and next 
        # operation associated with each condition. 
        # self.next_condi and self.next_event should have the same key. 
        self.next_condi = {}
        # 'default' is for when no condition is satisfied in the 
        # condition dictionary. 
        self.next_event = {'default': Event()}

    def append_condi(self, condition):
        if condition != 'else':
            condition, op, val = condition
            self.next_condi[condition] = (val, self._decode_op(op))
        self.next_event[condition] = Event()
    
    def _decode_op(self, op):
        '''
        given a logic opeation as string ('>', '<', '==', '>=', '<=')
        give the appropreiate dunction handle

        Args:
            op (str): can be one of '>', '<', '==', '>=', '<='
        
        returns 
            func (handle): equivalent operator as functions
        '''
        if op == '>': return operator.gt
        elif op == '<': return operator.lt
        elif op == '>=': return operator.ge
        elif op == '<=': return operator.le
        elif op == '==': return operator.eq
        elif op == '!=': return operator.ne


class RecipeObject: 

    def __init__(self, input_arg):
        self.input_arg = input_arg
        # Start with a empty event
        self.start = Event()
        self.tail = self.start

    def add_new(self, event):
        '''
        Add a event to the tail of current linked list. 

        Args: 
            event (Event): the event to be added, can be a operation or a condition
            condi (Tuple): in the form of (condi_type, condi_name, op, val)
        
        Returns: 
            None: In the case when the event is a operation, a None is returned
            Event: If the event is a condition, return the reference to the 
                   current tail to keep track of the beginning of this conditional statement.
                   In this case, self.tail is updated to the input event
        '''
        # save the current tail of the list to return
        tail = self.tail
        if self.start.ref == None:
            # The linked list is currently empty, so set the start of the list
            self.start.ref = event

        else:
            # The current tail is a operation, so just add the event to the next 
            self.tail.ref = event

        return tail

    def _search_list_DFS(self, func):
        '''
        Do a DFS with the current head as root
        Return as soon as func(node) == True

        --TODO--
        '''     
        
    def set_tail(self, tail):
        '''
        Set the current tail of the linked list tail to the specified 
        Condition class.

        Args:
            tail (Condition): the tail we wish to set the
        
        Returns: 
            success (bool): Only returns if True, else a ValueError os raised
                
        Note: 
            Only Condition class can be used to set as tail!
            The instance of the specified tail must be already part of the 
            list!
        '''
        
        # Do a depth-first search on the linked list to determine 
        # where the specified tail is
        # The root is the head of the list.

        try: 
            assert(type(tail) == Condition) 
        except AssertionError:
            raise ValueError('Expected "tail" to be Condition(),\
                                 instead got {}'.format(type(tail)))

        S = [] # here S is used as a stack
        S.append(self.start) # append() is equivalent to push()
                             # push root onto the stack
        while (len(S) != 0):
            node = S.pop()

            if node.type == 'op':
                # There is only one child for a operation node
                S.append(node.next)

            elif node.type == 'condi':
                # The children of a condition node is specified
                # in the next_event dictionary (next event)
                if node == tail: 
                    # node and tail are classes, and we are 
                    # comparing their address: are they the same instance?
                    # If so, then we have found our new tail
                    self.tail = node
                    return True
                # This is not the node we are looking for 
                # search its children in the linked list
                for _, event in node.next_event.items():
                    # If the event is None, then this node 
                    # is a leaf of the linked ist
                    if event != None:
                        S.append(event)
        # We have searched the entire list and we did not find the 
        # tail.
        raise ValueError('Could not find the provided Condition in list')
    
    def read_recipe(self, file):
        '''
        Given a input file path '.txt', read it as a recipe file.
        This parser enforce certain syntax in the recipe

        Args: 
            file (str): string of filename
        
        Returns
            success (bool): indicating whether the recipe is build successfuly
        '''
        module_name = 'prototypes.recipe_parser.primitives'
        tail_stack = [] # This is treated as a stack. 
                          # Keep track of how many nested conditions there are so far
        condi_stack = []  # This is also treated as a stack
                          # Whenever a 'ELSE IF' statement is detected, a condition statement 
                          # is pushed onto this stack
        with open(file, 'r') as f:
            # Will read the file line by line
            line = f.readline()
            while line:
                # Simiary to C++, we treat anything after '//' as comment 
                # so we ignore it.
                line = line[:line.find('//')].strip(' ')
                # Assume that a line is a command deperated by ' ' (blank space)
                # First command: Operation. Currently support 'DO', 'IF'
                # Second command: either a primitive name or a condition statement
                cmd = line.split(' ')
                try: 
                    if cmd[0] == 'DO':
                        # print('what?0')
                        # cmd[1] should be the name of the operation being imported
                        # Try to import the operation, and add it to the list
                        func = getattr(__import__(module_name, globals(), locals(), [cmd[1]]), \
                                        cmd[1])
                        oper = Operation(cmd[1], func)
                        self.add_new(oper)

                        self.tail = self.tail.ref.next
                

                    if cmd[0] == 'IF':
                        # print('what?1')
                        condi = Condition()
                        # 'if' statement is in the form "IF ( a < 1 ) {"
                        # we use {} to keep track of the if statement
                        # cmd[1], cmd[5] should be brackets ()
                        # cmd[2] should be a member of the argument class. 
                        # cmd[3] should be a operation of the form '>', etc
                        # cmd[4] should be an integer in str
                        # cmd[-1] should be first half of the '{}'
                        key = cmd[2]
                        condi.append_condi((cmd[2], cmd[3], int(cmd[4])))
                        tail_stack.append(('if', self.add_new(condi)))
                        self.tail = self.tail.ref.next_event[cmd[2]]
                

                    if cmd[0] == 'WHILE':
                        # print('what?2')
                        condi = Condition()
                        # 'WHILE' statments are similar to 'IF', but when the 
                        # {} closes, we want to evaluate the condition again 
                        # cmd[1], cmd[5] should be brackets ()
                        # cmd[2] should be a member of the argument class. 
                        # cmd[3] should be a operation of the form '>', etc
                        # cmd[4] should be an integer in str
                        # cmd[-1] should be first half of the '{}'
                        condi.append_condi((cmd[2], cmd[3], int(cmd[4])))
                        tail_stack.append(('while', self.add_new(condi)))
                        self.tail = self.tail.ref.next_event[cmd[2]]
                    
                    if cmd[0] == '}':
                        # This is a special line that signifies the end of 
                        # a condition statement.
                        # This line can present itself in three possible forms
                        #   "} ELSE IF ( a > 1 ) {"   (9 item)
                        #   "} ELSE {"                (3 item)
                        #   "}"                       (1 item)

                        # only pop when no 'else' statement follows
                        statement, tail = tail_stack[-1]
                        n_else = 1
                        if len(cmd) == 9:
                            # This can only happen if the top of the tail stack 
                            # is pushed by a 'IF' statement
                            # Add the new condition to the stack
                            # The next call of 'DO' should be called under this condition 
                            assert(statement == 'if')
                            # If the key already exists, then append a slightly 
                            # different key
                            if cmd[4] in tail.ref.next_condi.keys():
                                key = cmd[4] + '#' + str(n_else)
                                n_else += 1
                            tail.ref.append_condi((key, cmd[5], int(cmd[6])))
                            self.tail = tail.ref.next_event[key]
                        
                        elif len(cmd) == 3:
                            # This also can only happen if the top of the stack 
                            # is pushed by 'IF'
                            assert(statement == 'if')
                            # In this case, push a 'else' statement onto 
                            # the condition stack
                            tail.ref.append_condi('else')
                            self.tail = tail.ref.next_event['else']

                        
                        elif len(cmd) == 1:
                            # When this is the end of a 'IF' statement, we want to 
                            # link all the tail of other path in the linked list 
                            # to the next node, whether it be Condition() or Operation()
                            tail_stack.pop() 
                            if statement == 'if':
                                self.tail = tail.ref.next_event['default']

                            # When this is a end of a 'WHILE' statement, we want to 
                            # go back to the head and re-evaluate the condition 
                            if statement == 'while':
                                # Making sure that the current head is an Operation.
                                # The only possible case that the current head is not
                                # is when we have an empty while loop.

                                # Simply point the next field to the beginning of the 
                                # Condition. We have already done this
                                self.add_new(tail.ref)
                except: 
                    print('Could not read recipe file!: {}'.format(line))
                    print(sys.exc_info[1])
                line = f.readline()

    def execute(self, input_arg, receipt_path):
        head = self.start
        argument = input_arg
        receipt = open(receipt_path, 'w+')

        # If the current head has no operation (EOF), then 
        # terminate the loop and exit
        while(head.ref != None):
            # print('head = {}'.format(head.ref))
            # Check what the current event type is: 
            if head.ref.type == 'op':
                # Current head is a operation, so just executed the 
                # current operation and go to next
                # print('op.next = {}'.format(head.ref.next))
                argument = head.ref.call(argument)
                receipt.write(head.ref.name)
                receipt.write('\n')
                head = head.ref.next
            elif head.ref.type == 'condi':
                # Current head is a condition, so we need to check what  
                # previous argument evaluates to true
                if len(head.ref.next_condi) == 0:
                    # No condition. Simply go to default
                    head = head.ref.next_event['default']
                else: 
                    # There are conditions that specifies which operation comes next 
                    # loop through and check each condition, update operation accordingly
                    try:
                        # an 'else' event is present 
                        # This event is executed when all other condition 
                        # evaluates to False
                        new_head = head.ref.next_event['else']
                    except KeyError:
                        # No 'else' event, so just go to the default
                        new_head = head.ref.next_event['default']
                    for member, val in vars(argument).items():
                        for key in head.ref.next_condi.keys():
                            try:
                                if key.split('#')[0] == member: 
                                    cond = head.ref.next_condi[key]
                                    if cond[1](val, cond[0]):
                                        # Update the next operation
                                        new_head = head.ref.next_event[key]
                                        break # break out of the for loop
                            except KeyError: 
                                # The condition does not care about this member, skip it
                                pass
                    # for key, val in enumerate(head.ref.next_event.items()):
                    #     print('#', key, val[0], val[1].ref)
                    # print('condi.next = {}'.format(new_head.ref))
                    head = new_head
        return argument

if __name__ == '__main__':
    pass

            