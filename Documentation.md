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

It was mentioned before that a missing value is read from the default input. For example:

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

If the name cannot be read from the command parameters as text, its value becomes "World".

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

Edit _scratch.grid_ again:

```vb
[A1] := "Cookie Sales"
[A2:E2] := {"Sales Rep", "Region", "# Orders", "Total Sales", "Avg Order"}
[A3:D6] := {
  "Frank", "West", 268, 72707;
  "Harry", "North", 224, 41676;
  "Janet", "North", 286, 87858;
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
[^A3] := {
  new CookieSales with (SalesRep = "Frank", Region = "West", Orders = 268, Total = 72707),
  new CookieSales with (SalesRep = "Harry", Region = "North", Orders = 224, Total = 41676),
  new CookieSales with (SalesRep = "Janet", Region = "North", Orders = 286, Total = 87858),
  new CookieSales with (SalesRep = "Martha", Region = "East", Orders = 228, Total = 49017)}
Return [E3]
```

The hat `^` before an address indicates the top-left corner of a range.
The fields of the objects are distributed in the cells of a row, which means the grid contents remain the same as before.
