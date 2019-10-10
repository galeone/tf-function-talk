# Dissecting tf.function to discover AutoGraph strengths and subtleties
<!-- classes: title -->

<!-- note
Hello everyone!

In this talk, I'm going to show you how to design functions that can be correctly graph-converted using two of the most exciting new features of TensorFlow 2.0: AutoGraph and tf.function.

But first, let me introduce myself.
-->

<small><i>How to correctly write graph-convertible functions in TensorFlow 2.0.</i></small>

---

## About me

<!-- note
I'm Paolo Galeone, and I'm a computer engineer, I do Computer Vision and Machine Learning for a living and... I'm obsessed with TensorFlow.

I started using TensorFlow as soon as Google released it publicly, around November 2015, when I was a Research fellow at the computer vision laboratory of the  University of Bologna,

And I never stopped since then.

In fact, I blog about TensorFlow, I answer questions on StackOverflow about TensorFlow, I write opensource software with TensorFlow, and I use it every day at work.
For these reasons, Google awarded me with the title of Google Developer Expert in Machine Learning.

As I mentioned, I have a blog pgaleone.eu (**point to the address**) in which I write about TensorFlow mainly and I invite you to read it, especially because this talk is born from a three-part article I wrote about tf.function and autograph.

Ok, after this brief introduction, we are ready to start!
-->

![me](images/me_hk.jpg)

Computer engineer | Head of ML & CV @ ZURU Tech Italy | Machine Learning GDE

- Blog: https://pgaleone.eu/
- Github: [https://github.com/galeone/](galeone)
- Twitter: [@paolo_galeone](https://twitter.com/paolo_galeone)
- Author: [Hands-On Neural Networks with TensorFlow 2.0](https://amzn.to/2ZULPzh)

---

## TensorFlow 2.0 & DataFlow Graphs

<!-- sectionTitle: TensorFlow 2.0 & DataFlow Graphs -->
<!-- note
In TF 2.0 the concept of graph definition and session execution, the core of the descriptive way of programming used in TF 1.x, are disappeared, or better they have been hidden, in favor of the eager execution.

The eager execution is just the Python-like execution of the computation, line by line.
These new design choice has been made to lower the entry barriers, making TensorFlow more Pythonic and easy to use.

Of course, the description of the computation using DataFlow graphs, proper of TensorFlow 1.x, have too many advantages that TensorFlow 2.0 must still have. 

For instance, graphs give us:

- a faster execution speed;
- graphs are easy to replicate and distribute
- graphs are Language Agnostic Representation, in fact, a graph is not a Python program but a description of a computation; being agnostic to the language they can be generated using Python and used in any other programming language.
- Moreover, automatic differentiation comes almost for free when the computation is described using graphs.

So, to merge the graph advantages proper of TF1 and the ease of use of eager execution, TensorFlow introduced tf.function and AutoGraph.
-->

- Execution Speed
- Language Agnostic Representation
- Easy to replicate and distribute
- Automatic Differentiation

---

##  tf.function and AutoGraph

<!-- note
tf.function allows you to transform a subset of Python syntax into a portable, high-performance graph, with a simple function decoration.

As it can be seen from the function signature, tf.function uses autograph.

AutoGraph lets you write graph code using natural Python syntax. In particular, AutoGraph allows us to use Python control flow statements (if, else, for, while and so on) inside tf.function decorated functions, and it automatically converts them into the appropriate TensorFlow graph nodes.

However, in practice, what happens when a tf.function decorated function is called?
-->

```python
# tf.function signature: it is a decorator.
def function(func=None,
             input_signature=None,
             autograph=True,
             experimental_autograph_options=None)
```

<b>tf.function uses AutoGraph</b>

AutoGraph converts Python control flow statements into appropriate TensorFlow graph ops.

---

## tf.function and AutoGraph

<!-- note
As we can see from the diagram, eager execution is disabled for a function decorated with tf.function.

On the first call

1. The function is executed and traced. Being eager executed disabled every tf.method just defines a tf.Operation node that produces a tf.Tensor output, exactly like in TensorFlow 1.x.
2. At the same time, AutoGraph is used to detect Python constructs that can be converted to their graph equivalent (while → tf.while, for → tf.while, if → tf.cond, assert → tf.assert, …).
3. From the function trace + autograph, the graph representation is built. In order to preserve the execution order in the defined graph, tf.control_dependencies is automatically added after every statement, in order to condition the line i+1 on the execution of line i.
4. The tf.Graph object has now been built.

Based on the function name and the input parameters, a unique ID is created and associated with the graph. The graph is cached into a map: map[id] = graph.
Any function call will re-use the defined graph if the key matches.

Since tf.function is a decorator, it forces us to organize the code using functions.
Functions are the TF2 replacement for the session objects.

Now that we have a basic understanding of how tf.function works, we can start using it to see if everything goes as expected.
-->

![tf-execution](images/tf-execution.png)

---

## The problem

<!-- note
That is the multiplication of 2 constant matrix followed by the addition of a scalar variable,
-->

Given the **constant** matrices
<br />

![aX](images/Ax.png)
<br />

And the scalar **variable** ![b](images/b.png)
<br />

Compute ![Ax_b](images/Ax_b.png)

---

## TensorFlow 1.x solution

<!-- note
In TensorFlow 1.x we have to
- first **describe the computation** as a **graph**, inside a graph scope.
- then create a special node, with the only goal of initializing the variables
- then create a session object, that is the object that receives the description of the coputation and places it on the correct hardware
- and finally use the session object to run the computation and getting the result

in TensorFlow 2.0, thanks to the eager execution the solution of the problem becomes easier.
-->

```python
g = tf.Graph()
with g.as_default():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(y))
```

---

## TensorFlow 2.0 solution: eager execution

<!-- note
In fact, we only have to declare the constants and the variables, and the computation is executed directly, without the need to create a session.

In order to replicate the same behavior of the session execution, we write the code inside a function.

Executing the function has the same behavior of the previous session.run.

The only peculiarity here is that every `tf` operation, like tf.constant, tf.matmul and so on, produce a `tf.Tensor` object and not a Python native type or a numpy array.

Therefore, we have to extract from the tf.Tensor the Numpy representation by calling the numpy method.

We can call the function as many times we want, and it works like any other Python function.

All right then, since we declared the computation inside a function, we can try to convert it to its graph representation using tf.function.
-->

```python
def f():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    return y
print([f().numpy() for _ in range(10)])
```

<b>Every tf.* op, produces a `tf.Tensor` object</b>

---

## From eager function to tf.function

<!-- note
One might expect that since this function works in eager mode, we can convert it to its graph representation only by decorating it with tf.function.

Let's try and see what happens - I added a print statement and a tf.print statement that will help to clarify what happens here.
-->

```python
@tf.function
def f():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y

f()
```

---

## From eager function to tf.function

<!-- note
This is the first output we see on the console.

When the function is called, the process of graph creating starts.
At this stage, only the Python code is executed, and its execution is traced in order to collect the required data to build the graph.

The tf.print call is not evaluated as any other tf.* method since TensorFlow already knows everything about that statements and it can use them as they are to build the graph.

Moving forward we can see the second output.
-->

```text
PRINT:  Tensor("add:0", shape=(2, 2), dtype=float32)
```

---

## From eager function to tf.function

<!-- note
Wat. An exception:

"tf.function-decorated function tired to create variables on non-first call."

Ok, what's going on here?
The exception is a little bit misleading since we called this function once, but tf.function internally called it more than once.

As it is easy to understand, tf.function is complaining about the variable object.

This exception brings us to the first lesson.
-->

```text
ValueError: tf.function-decorated function tried to create variables on non-first call.
```

![wat](images/wat-cat.jpg)

---

## Lesson #1: functions that create a state

<!-- note
A `tf.Variable` object in eager mode is just a Python object that gets destroyed as soon as it goes out of scope.

... and that's why this function in eager mode works correctly.

But a `tf.Variable` in a tf.function-decorated function is the definition of a node in a persistent graph since eager execution is disabled in this context.

So, since the graph is persistent, we can't define a variable object every time we call a function.
-->

> A `tf.Variable` object in eager mode is just a Python object that gets destroyed as soon as it goes out of scope.

​​​

> A `tf.Variable` object in a tf.function-decorated function is the definition of a node in a persistent graph (eager execution disabled).

---

## The solution

<!-- note
So the solution to this problem is to think about the graph definition while designing the function.

Since we can't declare a new variable every time the function is called, we have to take care of this manually.

Declaring the variable as a private attribute of class F, and creating it only during the first call, we can correctly define a computational graph that works as we expect.

This brings us to the second lesson.
-->

```python
class F():
    def __init__(self):
        self._b = None

    @tf.function
    def __call__(self):
        a = tf.constant([[10, 10], [11., 1.]])
        x = tf.constant([[1., 0.], [0., 1.]])
        if self._b is None:
            self._b = tf.Variable(12.)
        y = tf.matmul(a, x) + self._b
        print("PRINT: ", y)
        tf.print("TF-PRINT: ", y)
        return y

f = F()
f()
```

---


## Lesson #2: eager function is not graph-convertible as is

<!-- note
There is no guarantee that Functions that work in eager mode are graph-convertible as they are.

Always define the function structure thinking about the graph that is being built.

OK! We can now move forward an analyze what happens when the input type of the function changes.
-->

> There is no 1:1 match between eager execution and the graph built by `@tf.function`.
> 
> Define the function thinking about the graph that is being built.

---

<!-- sectionTitle: Change the input type -->
# Change the input type

<!-- note
This part of the talk is of extreme importance since tf.function should bridge two completely different worlds.

In fact, Python is a dynamically-typed language where a function can accept any input type, while
TensorFlow, being a C++ library under the hood, it is strictly statically typed and every node n the graph must have a well-defined type.
-->

- **Python** is a dynamically-typed language
- **TensorFlow** is a strictly statically typed framework

---

## The function

<!-- note
This is the function we are going to use to understand how tf.function handles the Python typing system and its polymorphism.

On line 1, we can see that the function accepts a Python variable x that can be literally everything.
On line 2, we have the print function that is executed only once, during the function tracing.
On line 3: we have the tf.print function that is executed every time the graph is evaluated.
Line 4: x is returned.
-->

```python
@tf.function
def f(x):
    print("Python execution: ", x)
    tf.print("Graph execution: ", x)
    return x
```
<b>The function parameters type is used to create a graph, that is a statically typed object, and to assign it an ID</b>

---

## tf.Tensor as input

<!-- note
When the input is a tf.Tensor we expect that a graph is built for every different tf.Tensor dtype, and this should happen only once.

On every second call, thus, we don't want to see the line "Python execution" but only output of the graph execution.
-->

```python
print("##### float32 test #####")
a = tf.constant(1, dtype=tf.float32)
print("first call")
f(a)
a = tf.constant(1.1, dtype=tf.float32)
print("second call")
f(a)

print("##### uint8 test #####")

b = tf.constant(2, dtype=tf.uint8)
print("first call")
f(b)
b = tf.constant(3, dtype=tf.uint8)
print("second call")
f(b)
```

---

## tf.Tensor as input

<!-- note
et voilà, everything goes as we expect!

Since everything goes smoothly, we can deep dive a little bit more inside the autograph structure and check if the graph that is being built is the one we expect.

And of course, we expect a graph that contains only the graph execution.
-->

```text
##### float32 test #####
first call
Python execution:  Tensor("x:0", shape=(), dtype=float32)
Graph execution:  1
second call
Graph execution:  1.1

##### uint8 test #####
first call
Python execution:  Tensor("x:0", shape=(), dtype=uint8)
Graph execution:  2
second call
Graph execution:  3
```
<b>Everything goes as we expect</b>

---

## Inspecting the function

<!-- note
Using the autograph module, it is possible to see how autograph converts a Python function to its graph representation.

The code is, of course, hard to read since it's machine-generated, but we can notice 
something unexpected.

There is a reference to the Python execution inside the graph translation.
-->

```python
tf.autograph.to_code(f.python_function)
```

```python
def tf__f(x):
  try:
    with ag__.function_scope('f'):
      do_return = False
      retval_ = None
      with ag__.utils.control_dependency_on_returns(ag__.converted_call(print, None, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=ag__.Feature.ALL, internal_convert_user_code=True), ('Python execution: ', x), {})):
        tf_1, x_1 = ag__.utils.alias_tensors(tf, x)
        with ag__.utils.control_dependency_on_returns(ag__.converted_call('print', tf_1, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=ag__.Feature.ALL, internal_convert_user_code=True), ('Graph execution: ', x_1), {})):
          x_2 = ag__.utils.alias_tensors(x_1)
          do_return = True
          retval_ = x_1
          return retval_
  except:
    ag__.rewrite_graph_construction_error(ag_source_map__)
```

---

## Inspecting the function

<!-- note
Without digging too much in the code structure, we can see that there is the name of the function that is Python executed - print - it's arguments - "Python execution" comma x - wrapped inside a control dependency on return.

The second parameter of the autograph converted call method is the "owner" of the function print.

As can be seen, the owner is None, and this means there is no package known to autograph that contains the print function.

In short: this statement is converted to a no_op, it has no side effects and is only used by control dependencies to force the order of execution.

We can now go on and see what happens if the input is a Python native type and not a tf.Tensor object.
-->

```python
with ag__.utils.control_dependency_on_returns(
        ag__.converted_call(
            print, None, ag__.ConversionOptions(
                recursive=True,
                force_conversion=False,
                optional_features=ag__.Feature.ALL,
                internal_convert_user_code=True),
            ('Python execution: ', x), {})
        ):
```

<b>`converted_call` `owner` parameter is `None`: this line becomes a `tf.no_op()`</b>

---

## Python native type as input

<!-- note
The code is similar to the previous one; we just defined a helper function "printinfo" to be sure that we are feeding the correct data type.
Since the function is trivial, we expect the same behavior obtained before.
-->

```python
def printinfo(x):
  print("Type: ", type(x), " value: ", x)

print("##### int test #####")
print("first call")
a = 1
printinfo(a)
f(a)
print("second call")
b = 2
printinfo(b)
f(b)

print("##### float test #####")
print("first call")
a = 1.0
printinfo(a)
f(a)
print("second call")
b = 2.0
printinfo(b)
f(b)
```

---

## Python native type as input

<!-- note
Here we can see what happens when the integers are feed as input.

Something weird is going on since the "Python execution" line is displayed twice, and not only once as we expect.

The graph is begin recreated at every function invocation... that's weird. But things are getting even worse...
Let's have a look at the float execution.
-->

### Call with Python int

```text
##### int test #####
first call
Type:  <class 'int'>  value:  1
Python execution:  1
Graph execution:  1

second call
Type:  <class 'int'>  value:  2
Python execution:  2
Graph execution:  2
```

<b>The graph is being recreated at every function invocation</b>

---

## Python native type as input

<!-- note
The graph now is not being recreated at every invocation, but given a float input, we get an integer output.

But our function should return the input parameter x!
-->

### Call with Python float

```text
##### float test #####
first call
Type:  <class 'float'>  value:  1.0
Graph execution:  1
second call
Type:  <class 'float'>  value:  2.0
Graph execution:  2
```

- <b>The return type is WRONG.</b>
- <b>The graphs built for the integers 1 and 2 is reused for the float 1.0 and 2.0</b>

---

<!-- note
This behavior surprised me a lot, and I spent some time to figure out what's going on. What happens is summarized in the next lesson.
-->

![wat](images/wat-tall-cat.png)

---

## Lesson #3: no autoboxing

<!-- note
**read the lesson**.

This is a design choice I don't like about tf.function since it makes the graph conversion of a function not natural.
Moreover, since a new graph is being built for every different Python type, we also have the risk of designing terribly slow functions.
-->

> `tf.function` does not automatically convert a Python integer to a `tf.Tensor` with dtype `tf.int64`, and so on.
>
> The graph ID, when the input is not a `tf.Tensor` object, is built using the variable **value**.

That's why we used the same graph built for `f(1)` for `f(1.0)`, because `1 == 1.0`.

---

## No autoboxing: performance measurement

<!-- note
g is the identity function. In the first for loop, g is fed with `tf.Tensor` objects produced by the `tf.range` function execution.

The second for loop, instead, invoke `g` with 1000 different Python integers, and this means that we are building 1000 different graphs.

AutoGraph is highly optimized and works well when the input is a tf.Tensor object, while it creates a new graph for every different input parameter value with a huge drop in performance.

And this brings us to the 4th lesson.
-->

```python
@tf.function
def g(x):
  return x

start = time.time()
for i in tf.range(1000):
  g(i)
print("tf.Tensor time elapsed: ", (time.time()-start))

start = time.time()
for i in range(1000):
  g(i)
print("Native type time elapsed: ", (time.time()-start))
```

```text
tf.Tensor time elapsed:  0.41594886779785156
Native type time elapsed:  5.189513444900513
```

---

## Lesson #4: tf.Tensor everywhere

<!-- note
Use tf.Tensor everywhere, seriously.

tf.Tensor is not the only TensorFlow object that we have to use when using tf.function.

tf.function has this weird behavior when using Python native types, but it also has a weird behavior when using other Python native constructs.
-->

> Use `tf.Tensor` everywhere.

---

<!-- sectionTitle: Python operators -->
# Python Operators

<!-- note
This function works correctly in eager mode, given the tf.Tensor x that holds the constant value of 1, we expect to get the output "a == b", since a and b are the same Python object.

I hope we all agree that the final else should never be reached.

Ok, let's execute the function and see what happens.
-->

```python
@tf.function
def if_elif(a, b):
  if a > b:
    tf.print("a > b", a, b)
  elif a == b:
    tf.print("a == b", a, b)
  elif a < b:
    tf.print("a < b", a, b)
  else:
    tf.print("wat")
x = tf.constant(1)
if_elif(x,x)
```

---

## Python operators

<!-- note
wat.
Ok so for tf.function, a is not greater, equal, or lesser than b. How is this possible?
-->

<b>wat</b>

![wat dog](images/wat-dog.jpg)

---

## Problems

<!-- note
Keeping this short, there are several problems here. The bigger one that affects TensorFlow from the early releases is that the Python equal operator is not overloaded to use `tf.equal`.

The second huge problem is that AutoGraph handles the conversion of the if, elif and else statements but not the conversion of the boolean expressions defined using the Python built-in operators.
-->

- Python `__eq__` is not converted to `tf.equal` (even in eager mode!) but checks for the Python variable identity.
- AutoGraph converts the `if`, `elif`, `else` statements but
- AutoGraph does not converts the Python built-in operators (`__eq__`, `__lt__`, `__gt__`)

---

## Python operators

<!-- note
So, the correct way of writing this function is to use the TensorFlow boolean operators.
And this brings us to the last lesson.
-->

```python
@tf.function
def if_elif(a, b):
  if tf.math.greater(a, b):
    tf.print("a > b", a, b)
  elif tf.math.equal(a, b):
    tf.print("a == b", a, b)
  elif tf.math.less(a, b):
    tf.print("a < b", a, b)
  else:
    tf.print("wat")
```

---

## Lesson 5: operators
<!-- note
Use the TensorFlow operations everywhere, seriously.
-->

> Use the TensorFlow operators explicitly everywhere.

---

## Things are changing

<center><blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hey <a href="https://twitter.com/paolo_galeone?ref_src=twsrc%5Etfw">@paolo_galeone</a>, great blog post series on tf.function! I&#39;ve tried the if_elif_else case (from part 3: <a href="https://t.co/HukmaUY4dL">https://t.co/HukmaUY4dL</a>) this afternoon, and it looks like it has been fixed in 2.0.0rc0. Thought you might want to know</p>&mdash; Raphael Meudec (@raphaelmeudec) <a href="https://twitter.com/raphaelmeudec/status/1172510929659019264?ref_src=twsrc%5Etfw">September 13, 2019</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script></center>

The lessons presented, however, are all still valid: following them helps you writing idiomatic TensorFlow 2.0 code.

---

<!-- sectionTitle: Recap -->
<!-- note
OK we are reaching the end of the talks, so here it is a recap of what we learned.

**READ THE POINTS**
-->
# Recap

1. `tf.Variable` need a special treatment.
2. Think about the graph while designing the function: eager to graph is not straightforward.
3. There is no autoboxing of Python native types, therefore
4. Use `tf.Tensor` everywhere.
5. Use the TensorFlow operators explicitly everywhere.

---

<!-- note
The talk is finished, and I hope you enjoyed it. I just want to let you know that I'm authoring a book about TensorFlow and neural networks where I explain the TensorFlow ecosystem while designing neural networks-based applications.

If you enjoyed this talk and you want to get informed of new articles published on the blog, or when the book is out, just leave your email to the subscribe page.
-->

## Hands-On Neural Networks with TensorFlow 2.0

![book cover](images/cover.png)

Stay updated: https://pgaleone.eu/subscribe

---

<!-- sectionTitle: The End -->

# The End

<br />

Thank you :smile:

<br />

- Blog: https://pgaleone.eu/ (newsletter https://pgaleone.eu/subscribe)
- Twitter: https://twitter.com/paolo_galeone
- GitHub: https://github.com/galeone
