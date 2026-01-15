# The Grid Programming Language

The **Grid** programming language was created to handle tabular data and complex formulas.

## Previous Languages Called Grid

[Grid](https://gridlang.github.io/grid): _A high-level programming language with Javascript-like syntax. Named after its "grid" looping operator. (2024)_

[GRIDLang](https://github.com/akshaynagpal/GRIDLang): _A language to design games in an intuitive and expressive manner. It can be used to quickly prototype move-driven grid-based games. (2017)_

[Grid](https://esolangs.org/wiki/Grid): _Grid is not initially intended to be a programming language. However its computational properties are interesting and may be worth documenting. (2019)_

## First Program

Create a file called _helloworld.grid_ with the contents:

```vb
Return "Hello, World!"
```

The text is sent to the default output (the console).

```bash
> grid helloworld.grid
Hello, World!
```

You can try to output other things, for example a calculation:

```vb
Return (12 ^ 2) / 20
```

The result is 7.2.

## Inputs

Edit _helloworld.grid_ to add an input:

```vb
Input name
Return $"Hello, {name}!"
```

The name is read from the command parameters. If it is missing in the command parameters, it is read from the default input (keyboard entry).

```bash
> grid helloworld.grid Sam
Hello, Sam!
```

The dollar sign `$` before the quote indicates _string interpolation_, which just means you can use variables or expressions inside your text.

There can be multiple inputs, for example:

```vb
Input a, b
Return (a ^ 2) / b
```

## Type Constraints

The language has some predefined types: Text, Number and Logical.

Adding type annotations makes a program more robust.

Edit _helloworld.grid_ to add a type constraint for the input:

```vb
Input name as text
Return $"Hello, {name}!"
```

Note that **Grid** is case-insensitive.

## Default Value

A missing input value must be entered through the keyboard. For example:

```bash
> grid helloworld.grid
name: Gil
Hello, Gil!
```

That can be changed by providing a default value for the input. Edit _helloworld.grid_ again:

```vb
Input name as text or = "World"
Return $"Hello, {name}!"
```

Now if the name cannot be read from the command parameters as text, its value becomes "World".

```bash
> grid helloworld.grid
Hello, World!
```

## The Grid

The grid can be used as a scratch pad to store values or calculation results. Its columns are named from 'A' to 'Z', then 'AA' and so on. Its rows are numbered from 1 onwards.

Create a file called _scratch.grid_ with the contents:

```vb
[A1] := "Cookie Sales"
[A2] := "Sales Rep"
[B2] := "Region"
[C2] := "# Orders"
[D2] := "Total Sales"
[A3] := "Frank"
[B3] := "West"
[C3] := 268
[D3] := 72707
[A4] := "Harry"
[B4] := "North"
[C4] := 224
[D4] := 41676
[A5] := "Janet"
[B5] := "North"
[C5] := 286
[D5] := 87858
[A6] := "Martha"
[B6] := "East"
[C6] := 228
[D6] := 49017
```

That simply fills the grid with values. If you run the program, it won't output anything:

```bash
> grid scratch.grid
```

Edit _scratch.grid_ to add a calculation of the average order amount:

```vb
[A1] := "Cookie Sales"
[A2] := "Sales Rep"
[B2] := "Region"
[C2] := "# Orders"
[D2] := "Total Sales"
[A3] := "Frank"
[B3] := "West"
[C3] := 268
[D3] := 72707
[A4] := "Harry"
[B4] := "North"
[C4] := 224
[D4] := 41676
[A5] := "Janet"
[B5] := "North"
[C5] := 286
[D5] := 87858
[A6] := "Martha"
[B6] := "East"
[C6] := 228
[D6] := 49017
[E2] := "Avg Order"
[E3:E6] := @ [D] / [C]
Return [E3]
```

Two adresses separated with a colon `:` describe a _range_, and `@` before the formula means that the calculation should only use values from current row. It is the _implicit intersection_ operator.

The output value is 271.294776. You can also check other cell values.

## Arrays

Instead of entering the data cell by cell, the grid can be filled row by row.

Edit _scratch.grid_ as follows:

```vb
[A1] := "Cookie Sales"
[A2:E2] := {"Sales Rep", "Region", "# Orders", "Total Sales", "Avg Order"}
[A3:D3] := {"Frank", "West", 268, 72707}
[A4:D4] := {"Harry", "North", 224, 41676}
[A5:D5] := {"Janet", "North", 286, 87858}
[A6:D6] := {"Martha", "East", 228, 49017}
[E3:E6] := @ [D] / [C]
Return [E3]
```

A list of values inside braces `{}` forms an _array_. If the values are separated with a comma `,` it is a one-dimensional array.

If values are separated with a semicolon `;` a new row starts, and the array becomes two-dimensional.

Edit _scratch.grid_ to use a two-dimensional array:

```vb
[A1] := "Cookie Sales"
[A2:E2] := {"Sales Rep", "Region", "# Orders", "Total Sales", "Avg Order"}
[A3:D6] := { _
  "Frank", "West", 268, 72707; _
  "Harry", "North", 224, 41676; _
  "Janet", "North", 286, 87858; _
  "Martha", "East", 228, 49017}
[E3:E6] := @ [D] / [C]
Return [E3]
```

The output should stay the same as before.

## Custom Type

Let us define a custom type that describes each row of the table. Edit _scratch.grid_:

```vb
Define CookieSales as Type
  : SalesRep as Text
  : Region as Text
  : Orders as Number
  : Total as Number
  : Average = Total / Orders
End CookieSales

[A1] := "Cookie Sales"
[A2:E2] := {"Sales Rep", "Region", "# Orders", "Total Sales", "Avg Order"}
[^A3] := { _
  new CookieSales with (SalesRep = "Frank", Region = "West", Orders = 268, Total = 72707), _
  new CookieSales with (SalesRep = "Harry", Region = "North", Orders = 224, Total = 41676), _
  new CookieSales with (SalesRep = "Janet", Region = "North", Orders = 286, Total = 87858), _
  new CookieSales with (SalesRep = "Martha", Region = "East", Orders = 228, Total = 49017)}
Return [E3]
```

The hat `^` before an address indicates the top-left corner of a range.
The fields of each object are distributed in the cells of a row, which means the grid contents remain the same as before.

The average value is calculated using a formula. You can try adding more calculated fields to the type.

## More Constraints

We can declare the type of a variable, but we could be even more specific.

Edit _scratch.grid_ to give a unit for the sales total and specify that it is always positive:

```vb
Define CookieSales as Type
  : SalesRep as Text
  : Region as Text
  : Orders as Number
  : Total as Number of Dollar >= 0
  : Average = Total / Orders
End CookieSales

[A1] := "Cookie Sales"
[A2:E2] := {"Sales Rep", "Region", "# Orders", "Total Sales", "Avg Order"}
[^A3] := { _
  new CookieSales with (SalesRep = "Frank", Region = "West", Orders = 268, Total = 72707), _
  new CookieSales with (SalesRep = "Harry", Region = "North", Orders = 224, Total = 41676), _
  new CookieSales with (SalesRep = "Janet", Region = "North", Orders = 286, Total = 87858), _
  new CookieSales with (SalesRep = "Martha", Region = "East", Orders = 228, Total = 49017)}
Return [E3]
```

Note that two numbers must use the same unit if you want to add them together.

If you like, you can define a custom type to avoid typing the same constraints every time:

```vb
Define Credit as Type(Number) of Dollar >= 0
End Credit
```

Now 'Credit' can be used instead of 'Number'.

## Named Variables

The grid is not the only place that can hold values.

Create a new file called _variables.grid_ with the contents:

```vb
: result = $"The result is {n - value}"
For n init 5
Let value = 2
Return result
```

That defines three named variables, _result_, _n_ and _value_ using three different ways.

There is not one way better than the others. All have their own merits.

For example, _n_ is assigned a value using **init** which means it can receive a new value later.

## Updating a Variable With Push

Edit _variables.grid_ to update _n_ with a new value:

```vb
: result = $"The result is {n - value}"
For n init 5
Let value = 2
Push n = n * 2.2
Return result
```

The **push** instruction is used to update a variable and all the variables that depend on it.

```bash
> grid variables.grid
The result is 9
```

Try to output the result before **push**:

```vb
: result = $"The result is {n - value}"
For n init 5
Let value = 2
Return result
Push n = n * 2.2
Return result
```

When **return** is used multiple times it gives multiple outputs.

```bash
> grid variables.grid
The result is 3
The result is 9
```

Note how _result_ which depends on _n_ is being updated by `Push n`. A **Grid** program can contain **push** instructions that affect previously defined variables.

That property is essential to keep named variables and the grid consistent over time.
