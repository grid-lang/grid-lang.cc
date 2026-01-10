# The Grid Programming Language

The **Grid** programming language was created to handle tabular data and complex formulas.

## Previous Languages Called Grid

[Grid](https://gridlang.github.io/grid):

A high-level programming language with Javascript-like syntax. Named after its "grid" looping operator. (2024)

[GRIDLang](https://github.com/akshaynagpal/GRIDLang):

A language to design games in an intuitive and expressive manner. It can be used to quickly prototype move-driven grid-based games. (2017)

[Grid](https://esolangs.org/wiki/Grid):

Grid is not initially intended to be a programming language. However its computational properties are interesting and may be worth documenting. (2019)

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

If name cannot be read from the command parameters as text, its value becomes "World".

```bash
> grid helloworld.grid
Hello, World!
```
