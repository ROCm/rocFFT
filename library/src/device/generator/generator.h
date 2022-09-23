// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <algorithm>
#include <any>
#include <iostream>
#include <numeric>
#include <optional>
#include <string.h>
#include <string>
#include <variant>
#include <vector>

//
// Helpers
//

template <typename T>
std::string vrender(const T& x)
{
    return std::visit([](const auto a) { return a.render(); }, x);
}

template <typename T>
unsigned int get_precedence(const T& x)
{
    return std::visit([](const auto a) { return a.precedence; }, x);
}

// We have a circular dependency here when trying to use std::variant
// for our abstract syntax tree, e.g.:
//
// - Expression variants contains classes like Variable, Literal,
//   Add, Subtract
// - Those classes can themselves contain Expressions
//
// But, std::variant and std::vector can both use incomplete types,
// so long as we don't call any methods on the containers while
// the types are still incomplete.
//
// So we can resolve this:
// - forward-declare all classes that can go into the variant
// - declare the variant type in terms of the classes
// - declare all of the classes
//   - if they require Expression members, put them in a std::vector
//   - don't implement any method bodies that would call std::vector methods
//     (including constructors)
// - after all classes are declared, implement all remaining class methods

enum class Component
{
    REAL,
    IMAG,
    BOTH,
};

//
// Expressions
//

struct ScalarVariable;
class Variable;
class Literal;
class ComplexLiteral;

class Add;
class Subtract;
class Multiply;
class ComplexMultiply;
class Divide;
class Modulus;

class ShiftLeft;
class ShiftRight;
class And;
class BitAnd;
class Or;
class Less;
class LessEqual;
class Greater;
class GreaterEqual;
class Equal;
class NotEqual;

class UnaryMinus;
class Not;
class PreIncrement;
class PreDecrement;

class Ternary;

// FFT expressions

class LoadGlobal;

class TwiddleMultiply;
class TwiddleMultiplyConjugate;

class Parens;

class CallExpr;

class IntrinsicLoad;

using Expression = std::variant<ScalarVariable,
                                Variable,
                                Literal,
                                ComplexLiteral,
                                Add,
                                Subtract,
                                Multiply,
                                ComplexMultiply,
                                Divide,
                                Modulus,
                                ShiftLeft,
                                ShiftRight,
                                And,
                                BitAnd,
                                Or,
                                Less,
                                LessEqual,
                                Greater,
                                GreaterEqual,
                                Equal,
                                NotEqual,
                                UnaryMinus,
                                Not,
                                PreIncrement,
                                PreDecrement,
                                Ternary,
                                LoadGlobal,
                                TwiddleMultiply,
                                TwiddleMultiplyConjugate,
                                Parens,
                                CallExpr,
                                IntrinsicLoad>;

class OptionalExpression
{
    std::any expr;

public:
    OptionalExpression(){};
    explicit OptionalExpression(const Expression& expr);
    OptionalExpression& operator=(const Expression& in_expr);
    Expression          operator*() const;
                        operator bool() const;
};

class Literal
{

public:
    static const unsigned int precedence = 0;

    Literal(int num)
        : value(std::to_string(num))

    {
    }
    Literal(unsigned int num)
        : value(std::to_string(num))
    {
    }
    Literal(const std::string& val)
        : value(val)
    {
    }
    Literal(const char* val)
        : value(val)
    {
    }

    std::string value;

    std::string render() const
    {
        return value;
    }
};

struct ScalarVariable
{
    static const unsigned int precedence = 0;
    std::string               name, type;
    Component                 component;
    OptionalExpression        index;

    ScalarVariable(std::string name, std::string type, Component component = Component::BOTH)
        : name(name)
        , type(type)
        , component(component){};

    std::string render() const;
};

class Variable
{
public:
    static const unsigned int precedence = 0;
    std::string               name, type;
    bool                      pointer = false, restrict = false;
    ScalarVariable            x, y;
    Component                 component;
    OptionalExpression        index;
    OptionalExpression        size;
    // default value for argument and template declarations
    OptionalExpression decl_default;

    Variable(const std::string& _name,
             const std::string& _type,
             bool               pointer = false,
             bool restrict              = false,
             unsigned int size          = 0);

    Variable(const std::string& _name,
             const std::string& _type,
             bool               pointer,
             bool restrict,
             const Expression& _size);

    Variable(const ScalarVariable& v)
        : name(v.name)
        , type(v.type)
        , x(v.name + ".x", v.type)
        , y(v.name + ".y", v.type){};

    Variable(const Variable& v);
    Variable(const Variable& v, const Expression& _index);

    Variable&      operator=(const Variable&) = default;
    Variable       operator[](const Expression& index) const;
    ScalarVariable address() const;

    std::string render() const;
};

class ArgumentList
{
public:
    ArgumentList(){};
    ArgumentList(const std::initializer_list<Variable>& il)
        : arguments(il){};
    ArgumentList(const std::vector<Variable>& arguments)
        : arguments(arguments){};

    std::vector<Variable> arguments;
    std::string           render() const;
    std::string           render_decl() const;
                          operator bool() const;
    void                  append(Variable);

    // find an argument with the specified name and set it to the
    // supplied value
    void set_value(const std::string& name, const std::string& value);
};

using TemplateList = ArgumentList;

std::string ArgumentList::render() const
{
    std::string f;
    if(!arguments.empty())
    {
        f = arguments[0].render();
        for(unsigned int i = 1; i < arguments.size(); ++i)
        {
            f += ",";
            f += arguments[i].render();
        }
    }
    return f;
}

ArgumentList::operator bool() const
{
    return !arguments.empty();
}

void ArgumentList::append(Variable v)
{
    arguments.push_back(v);
}

void ArgumentList::set_value(const std::string& name, const std::string& value)
{
    for(auto& arg : arguments)
    {
        if(arg.name == name)
        {
            arg.name = value;
            return;
        }
    }
    // didn't find the argument - that should be a programmer error
    throw std::runtime_error("ArgumentList::set_value failed to find " + name);
}

class CallExpr
{
public:
    static const unsigned int precedence = 0;

    std::string             name;
    TemplateList            templates;
    std::vector<Expression> arguments;

    CallExpr(const std::string& name, const std::vector<Expression>& arguments);
    CallExpr(const std::string&             name,
             const TemplateList&            templates,
             const std::vector<Expression>& arguments);

    std::string render() const;
};

class ComplexMultiply
{
public:
    static const unsigned int precedence = 5;
    explicit ComplexMultiply(const std::vector<Expression>& args)
        : args(args)
    {
    }

    std::string render() const;

    std::vector<Expression> args;
};

class Ternary
{
public:
    static const unsigned int precedence = 16;
    Ternary(const Expression& cond, const Expression& true_result, const Expression& false_result);
    explicit Ternary(const std::vector<Expression>& args);
    std::string render() const;

    std::vector<Expression> args;
};

class LoadGlobal
{
public:
    static const unsigned int precedence = 18;
    LoadGlobal(const Expression& ptr, const Expression& index);
    explicit LoadGlobal(const std::vector<Expression>& args);

    std::string render() const;

    std::vector<Expression> args;
};

class TwiddleMultiply
{
public:
    static const unsigned int precedence = 18;
    TwiddleMultiply(const Variable& a, const Variable& b)
        : a(a)
        , b(b)
    {
    }
    Variable    a;
    Variable    b;
    std::string render() const;
};

class TwiddleMultiplyConjugate
{
public:
    static const unsigned int precedence = 18;
    TwiddleMultiplyConjugate(const Variable& a, const Variable& b)
        : a(a)
        , b(b)
    {
    }
    Variable    a;
    Variable    b;
    std::string render() const;
};

class Parens
{
public:
    static const unsigned int precedence = 0;
    explicit Parens(const Expression& inside);
    explicit Parens(const std::vector<Expression>& args);

    std::vector<Expression> args;
    std::string             render() const;
};

class IntrinsicLoad
{
public:
    static const unsigned int precedence = 18;
    explicit IntrinsicLoad(const std::vector<Expression>& args);

    // data, voffset, soffset, rw
    std::vector<Expression> args;
    std::string             render() const;
};

#define MAKE_OPER(NAME, OPER, PRECEDENCE)                           \
    class NAME                                                      \
    {                                                               \
        std::string oper{OPER};                                     \
                                                                    \
    public:                                                         \
        static const unsigned int precedence = PRECEDENCE;          \
        std::vector<Expression>   args;                             \
        explicit NAME(const std::initializer_list<Expression>& il); \
        explicit NAME(const std::vector<Expression>& il);           \
        std::string render() const;                                 \
    };

#define CONSTRUCT_OPER(NAME)                                \
    NAME::NAME(const std::initializer_list<Expression>& il) \
        : args(il){};                                       \
    NAME::NAME(const std::vector<Expression>& il)           \
        : args(il){};

#define MAKE_BINARY_METHODS(NAME)                                  \
    std::string NAME::render() const                               \
    {                                                              \
        std::string s;                                             \
        if(get_precedence(args[0]) > precedence)                   \
            s += "(" + vrender(args[0]) + ")";                     \
        else                                                       \
            s += vrender(args[0]);                                 \
        for(auto arg = args.begin() + 1; arg != args.end(); ++arg) \
        {                                                          \
            s += oper;                                             \
            if(get_precedence(*arg) >= precedence)                 \
                s += "(" + vrender(*arg) + ")";                    \
            else                                                   \
                s += vrender(*arg);                                \
        }                                                          \
        return s;                                                  \
    }

#define MAKE_UNARY_PREFIX_METHODS(NAME)               \
    std::string NAME::render() const                  \
    {                                                 \
        std::string s = oper;                         \
        if(get_precedence(args.front()) > precedence) \
            s += "(" + vrender(args.front()) + ")";   \
        else                                          \
            s += vrender(args.front());               \
        return s;                                     \
    }

MAKE_OPER(Add, " + ", 6);
MAKE_OPER(Subtract, " - ", 6);
MAKE_OPER(Multiply, " * ", 5);
MAKE_OPER(Divide, " / ", 5);
MAKE_OPER(Modulus, " % ", 5);

MAKE_OPER(Less, " < ", 9);
MAKE_OPER(LessEqual, " <= ", 9);
MAKE_OPER(Greater, " > ", 9);
MAKE_OPER(GreaterEqual, " >= ", 9);
MAKE_OPER(Equal, " == ", 10);
MAKE_OPER(NotEqual, " != ", 10);
MAKE_OPER(ShiftLeft, " << ", 7);
MAKE_OPER(ShiftRight, " >> ", 7);
MAKE_OPER(And, " && ", 14);
MAKE_OPER(BitAnd, " & ", 14);
MAKE_OPER(Or, " || ", 15);

MAKE_OPER(UnaryMinus, " -", 3);
MAKE_OPER(Not, " !", 3);
MAKE_OPER(PreIncrement, " ++", 3);
MAKE_OPER(PreDecrement, " --", 3);

MAKE_OPER(ComplexLiteral, ",", 17);

// end of Expression class declarations

CONSTRUCT_OPER(Add);
CONSTRUCT_OPER(Subtract);
CONSTRUCT_OPER(Multiply);
CONSTRUCT_OPER(Divide);
CONSTRUCT_OPER(Modulus);

CONSTRUCT_OPER(Less);
CONSTRUCT_OPER(LessEqual);
CONSTRUCT_OPER(Greater);
CONSTRUCT_OPER(GreaterEqual);
CONSTRUCT_OPER(Equal);
CONSTRUCT_OPER(NotEqual);
CONSTRUCT_OPER(ShiftLeft);
CONSTRUCT_OPER(ShiftRight);
CONSTRUCT_OPER(And);
CONSTRUCT_OPER(BitAnd);
CONSTRUCT_OPER(Or);

CONSTRUCT_OPER(UnaryMinus);
CONSTRUCT_OPER(Not);
CONSTRUCT_OPER(PreIncrement);
CONSTRUCT_OPER(PreDecrement);

CONSTRUCT_OPER(ComplexLiteral);

// TODO: use the standard binary method for Add when we no longer
// need to generate identical source to the python generator.
//
//MAKE_BINARY_METHODS(Add);
std::string Add::render() const
{
    std::string s;
    const char* render_oper = "";
    if(std::holds_alternative<Variable>(args[0]))
    {
        auto& var = std::get<Variable>(args[0]);
        // render compatibly with python generator if we're just
        // doing pointer math (i.e. &foo[bar] instead of foo + bar)
        if(!var.index && (var.pointer || var.size))
            return "&" + vrender(args[0]) + "[" + vrender(args[1]) + "]";
    }
    for(auto& arg : args)
    {
        s += render_oper;
        if(get_precedence(arg) > precedence)
            s += "(" + vrender(arg) + ")";
        else
            s += vrender(arg);
        render_oper = oper.c_str();
    }
    return s;
}

MAKE_BINARY_METHODS(Subtract);
MAKE_BINARY_METHODS(Multiply);
MAKE_BINARY_METHODS(Divide);
MAKE_BINARY_METHODS(Modulus);

MAKE_BINARY_METHODS(Less);
MAKE_BINARY_METHODS(LessEqual);
MAKE_BINARY_METHODS(Greater);
MAKE_BINARY_METHODS(GreaterEqual);
MAKE_BINARY_METHODS(Equal);
MAKE_BINARY_METHODS(NotEqual);
MAKE_BINARY_METHODS(ShiftLeft);
MAKE_BINARY_METHODS(ShiftRight);
MAKE_BINARY_METHODS(And);
MAKE_BINARY_METHODS(BitAnd);
MAKE_BINARY_METHODS(Or);

MAKE_UNARY_PREFIX_METHODS(UnaryMinus);
MAKE_UNARY_PREFIX_METHODS(Not);
MAKE_UNARY_PREFIX_METHODS(PreIncrement);
MAKE_UNARY_PREFIX_METHODS(PreDecrement);

Ternary::Ternary(const Expression& cond,
                 const Expression& true_result,
                 const Expression& false_result)
    : args{cond, true_result, false_result}
{
}

Ternary::Ternary(const std::vector<Expression>& args)
    : args(args)
{
}

std::string Ternary::render() const
{
    return vrender(args[0]) + " ? " + vrender(args[1]) + " : " + vrender(args[2]);
}

LoadGlobal::LoadGlobal(const Expression& ptr, const Expression& index)
    : args{ptr, index}
{
}

LoadGlobal::LoadGlobal(const std::vector<Expression>& args)
    : args(args)
{
}

std::string LoadGlobal::render() const
{
    return "load_cb(" + vrender(args[0]) + "," + vrender(args[1]) + ", load_cb_data, nullptr)";
}

std::string ScalarVariable::render() const
{
    return name;
}

std::string ArgumentList::render_decl() const
{
    std::string f;
    const char* separator = "";
    const char* comma     = ",";
    for(const auto& arg : arguments)
    {
        f += separator;
        f += arg.type;
        // arrays (i.e. where size != 0) are passed as pointers
        if(arg.pointer || arg.size)
            f += "*";
        if(arg.restrict)
            f += " __restrict__";
        f += " " + arg.name;
        if(arg.decl_default)
            f += " = " + vrender(*arg.decl_default);
        separator = comma;
    }
    return f;
}

Variable::Variable(const std::string& _name,
                   const std::string& _type,
                   bool               pointer,
                   bool restrict,
                   unsigned int size)
    : name(_name)
    , type(_type)
    , pointer(pointer)
    , restrict(restrict)
    , x(_name, _type, Component::REAL)
    , y(_name, _type, Component::IMAG)
    , component(Component::BOTH)
{
    if(size > 0)
        this->size = Expression{size};
}

Variable::Variable(const std::string& _name,
                   const std::string& _type,
                   bool               pointer,
                   bool restrict,
                   const Expression& _size)
    : name(_name)
    , type(_type)
    , pointer(pointer)
    , restrict(restrict)
    , x(_name, _type, Component::REAL)
    , y(_name, _type, Component::IMAG)
    , component(Component::BOTH)
    , size(_size)
{
}

// NOTE: cppcheck doesn't realize all of the members are actually
// initialized here
//
// cppcheck-suppress uninitMemberVar
Variable::Variable(const Variable& v)
    : name(v.name)
    , type(v.type)
    , pointer(v.pointer)
    , restrict(v.restrict)
    , x(v.name + ".x", v.type, Component::REAL)
    , y(v.name + ".y", v.type, Component::IMAG)
    , component(v.component)
    , index(v.index)
    , size(v.size)
    , decl_default(v.decl_default)
{
    if(index)
    {
        x.name = v.name + "[" + vrender(*index) + "].x";
        y.name = v.name + "[" + vrender(*index) + "].y";
    }
}

Variable::Variable(const Variable& v, const Expression& _index)
    : name(v.name)
    , type(v.type)
    , pointer(v.pointer)
    , restrict(v.restrict)
    , x(v.name, v.type, Component::REAL)
    , y(v.name, v.type, Component::IMAG)
    , component(v.component)
    , index(_index)
{
    size         = v.size;
    decl_default = v.decl_default;
    x.name       = v.name + "[" + vrender(*index) + "].x";
    y.name       = v.name + "[" + vrender(*index) + "].y";
}

ScalarVariable Variable::address() const
{
    if(index)
    {
        return ScalarVariable("&" + name + "[" + vrender(*index) + "]", type + "*");
    }
    return ScalarVariable("&" + name, type + "*");
}

std::string Variable::render() const
{
    if(index)
    {
        return name + "[" + vrender(*index) + "]";
    }
    return name;
}

Variable Variable::operator[](const Expression& index) const
{
    return Variable(*this, index);
}

Add operator+(const Expression& a, const Expression& b)
{
    return Add{a, b};
}

Subtract operator-(const Expression& a, const Expression& b)
{
    return Subtract{a, b};
}

Multiply operator*(const Expression& a, const Expression& b)
{
    return Multiply{a, b};
}

Divide operator/(const Expression& a, const Expression& b)
{
    return Divide{a, b};
}

Modulus operator%(const Expression& a, const Expression& b)
{
    return Modulus{a, b};
}

Less operator<(const Expression& a, const Expression& b)
{
    return Less{a, b};
}

LessEqual operator<=(const Expression& a, const Expression& b)
{
    return LessEqual{a, b};
}

Greater operator>(const Expression& a, const Expression& b)
{
    return Greater{a, b};
}

GreaterEqual operator>=(const Expression& a, const Expression& b)
{
    return GreaterEqual{a, b};
}

Equal operator==(const Expression& a, const Expression& b)
{
    return Equal{a, b};
}

NotEqual operator!=(const Expression& a, const Expression& b)
{
    return NotEqual{a, b};
}

ShiftLeft operator<<(const Expression& a, const Expression& b)
{
    return ShiftLeft{a, b};
}

ShiftRight operator>>(const Expression& a, const Expression& b)
{
    return ShiftRight{a, b};
}

And operator&&(const Expression& a, const Expression& b)
{
    return And{a, b};
}

BitAnd operator&(const Expression& a, const Expression& b)
{
    return BitAnd{a, b};
}

Or operator||(const Expression& a, const Expression& b)
{
    return Or{a, b};
}

UnaryMinus operator-(const Expression& a)
{
    return UnaryMinus{a};
}

Not operator!(const Expression& a)
{
    return Not{a};
}

PreIncrement operator++(const Expression& a)
{
    return PreIncrement{a};
}

PreDecrement operator--(const Expression& a)
{
    return PreDecrement{a};
}

OptionalExpression::operator bool() const
{
    return expr.has_value();
}

Expression OptionalExpression::operator*() const
{
    return std::any_cast<Expression>(expr);
}

OptionalExpression::OptionalExpression(const Expression& expr)
{
    this->expr = expr;
}
OptionalExpression& OptionalExpression::operator=(const Expression& in_expr)
{
    this->expr = in_expr;
    return *this;
}

std::string ComplexLiteral::render() const
{
    std::string ret       = "{";
    const char* separator = nullptr;
    for(const auto& arg : args)
    {
        if(separator)
            ret += separator;
        ret += vrender(arg);
        separator = oper.c_str();
    }
    ret += "}";
    return ret;
}

std::string ComplexMultiply::render() const
{
    auto a = std::get<Variable>(args[0]);
    auto b = std::get<Variable>(args[1]);
    auto r = ComplexLiteral{a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y};
    return r.render();
}

std::string TwiddleMultiply::render() const
{
    return ComplexLiteral{a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y}.render();
}

std::string TwiddleMultiplyConjugate::render() const
{
    return ComplexLiteral{a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y}.render();
}

Parens::Parens(const Expression& inside)
    : args{inside}
{
}

Parens::Parens(const std::vector<Expression>& args)
    : args{args}
{
}

std::string Parens::render() const
{
    return "(" + vrender(args.front()) + ")";
}

CallExpr::CallExpr(const std::string& name, const std::vector<Expression>& arguments)
    : name(name)
    , arguments(arguments){};

CallExpr::CallExpr(const std::string&             name,
                   const TemplateList&            templates,
                   const std::vector<Expression>& arguments)
    : name(name)
    , templates(templates)
    , arguments(arguments){};

std::string CallExpr::render() const
{
    std::string f;
    f += name;
    const char* separator = nullptr;
    const char* comma     = ",";
    if(!templates.arguments.empty())
    {
        f += "<";
        // template args just have the names, not types
        for(const auto& arg : templates.arguments)
        {
            if(separator)
                f += separator;
            f += arg.name;
            separator = comma;
        }
        f += ">";
    }
    f += "(";
    separator = nullptr;
    for(const auto& arg : arguments)
    {
        if(separator)
            f += separator;
        f += vrender(arg);
        separator = comma;
    }
    f += ")";
    return f;
}

IntrinsicLoad::IntrinsicLoad(const std::vector<Expression>& args)
    : args(args)
{
}

std::string IntrinsicLoad::render() const
{
    // intrinsic_load(const T* data, unsigned int voffset, unsigned int soffset, bool rw)
    return "intrinsic_load(" + vrender(args[0]) + "," + vrender(args[1]) + "," + vrender(args[2])
           + "," + vrender(args[3]) + ")";
}

//
// Statements
//

// Statements also have a circular dependency as described above for
// Expressions, for some classes.

class Assign;
class Call;
class CallbackDeclaration;
class Declaration;
class LDSDeclaration;
class For;
class While;
class If;
class ElseIf;
class Else;
class StoreGlobal;
class StoreGlobalPlanar;
class StatementList;
class Butterfly;
class IntrinsicStore;
class IntrinsicLoadToDest;

struct LineBreak
{
    static std::string render()
    {
        return "\n\n";
    }
};

struct SyncThreads
{
    static std::string render()
    {
        return "__syncthreads();";
    }
};

struct Return
{
    static std::string render()
    {
        return "return;\n";
    }
};

struct Break
{
    static std::string render()
    {
        return "break;\n";
    }
};

struct CommentLines
{
    std::vector<std::string> comments;
    std::string              render() const
    {
        std::string s;

        static const char* NEWLINE   = "\n";
        const char*        separator = "";
        for(auto c : comments)
        {
            s += separator;
            s += "// " + c;
            separator = NEWLINE;
        }
        return s;
    }
    explicit CommentLines(std::initializer_list<std::string> il)
        : comments(il){};
};

using Statement = std::variant<Assign,
                               Call,
                               CallbackDeclaration,
                               CommentLines,
                               Declaration,
                               LDSDeclaration,
                               For,
                               While,
                               If,
                               ElseIf,
                               Else,
                               StoreGlobal,
                               StoreGlobalPlanar,
                               LineBreak,
                               Return,
                               Break,
                               SyncThreads,
                               Butterfly,
                               IntrinsicStore,
                               IntrinsicLoadToDest>;

class Assign
{
public:
    Variable    lhs;
    Expression  rhs;
    std::string oper;

    Assign(const Variable& lhs, const Expression& rhs, const std::string& oper = "=")
        : lhs(lhs)
        , rhs(rhs)
        , oper(oper){};

    std::string render() const
    {
        return lhs.render() + " " + oper + " " + vrender(rhs) + ";";
    }
};

// +=, *= operators are just Assign with another operator
Assign AddAssign(const Variable& lhs, const Expression& rhs)
{
    return Assign(lhs, rhs, "+=");
}
Assign MultiplyAssign(const Variable& lhs, const Expression& rhs)
{
    return Assign(lhs, rhs, "*=");
}

//
// Declarations
//

class Declaration
{
public:
    Variable                  var;
    std::optional<Expression> value;
    explicit Declaration(const Variable& v)
        : var(v){};
    Declaration(const Variable& v, const Expression& val)
        : var(v)
        , value(val){};
    std::string render() const;
};

std::string Declaration::render() const
{
    std::string s;
    s = var.type;
    if(var.pointer)
        s += "*";
    s += " " + var.name;
    if(var.size)
        s += "[" + vrender(*var.size) + "]";
    if(value)
        s += " = " + vrender(*value);
    s += ";";
    return s;
}

class LDSDeclaration
{
public:
    explicit LDSDeclaration(const std::string& scalar_type)
        : scalar_type(scalar_type){};
    std::string scalar_type;
    std::string render() const
    {
        // Declare an LDS buffer whose size is defined at launch time.
        // The declared buffer is of type unsigned char, but is aligned
        // to a complex unit.
        //
        // We then define pointers to that buffer with real and
        // complex types, since the body of the function may look at
        // LDS as real values or as complex values (code for both is
        // generated, and we choose one at compile time via a template
        // parameter).
        //
        // TODO: Ideally we would use C++11 "alignas" and "alignof"
        // for alignment, but they're incompatible with "extern
        // __shared__".  Another alternative would be the __align__
        // HIP macro, but that is not currently present in hipRTC.
        return "extern __shared__ unsigned char __attribute__((aligned(sizeof(" + scalar_type
               + ")))) lds_uchar[];\nreal_type_t<" + scalar_type
               + ">* __restrict__ lds_real = reinterpret_cast<real_type_t<" + scalar_type
               + ">*>(lds_uchar);\n" + scalar_type
               + "* __restrict__ lds_complex = reinterpret_cast<" + scalar_type + "*>(lds_uchar);";
    }
};

class CallbackDeclaration
{
public:
    CallbackDeclaration(const std::string& scalar_type, const std::string& cbtype)
        : scalar_type(scalar_type)
        , cbtype(cbtype){};
    std::string scalar_type;
    std::string cbtype;
    std::string render() const
    {
        return "auto load_cb = get_load_cb<" + scalar_type + ", " + cbtype + ">(load_cb_fn);\n"
               + "auto store_cb = get_store_cb<" + scalar_type + ", " + cbtype
               + ">(store_cb_fn);\n";
    }
};

class Call
{
public:
    Call(const std::string& name, const std::vector<Expression>& arguments)
        : expr(name, arguments)
    {
    }
    Call(const std::string&             name,
         const TemplateList&            templates,
         const std::vector<Expression>& arguments)
        : expr(name, templates, arguments)
    {
    }

    CallExpr expr;

    std::string render() const
    {
        return expr.render() + ";";
    }
};

class StatementList
{
public:
    std::vector<Statement> statements;
    StatementList();
    StatementList(const std::initializer_list<Statement>& il);
    std::string render() const;
};

class For
{
public:
    Variable      var;
    Expression    initial;
    Expression    condition;
    Expression    increment;
    StatementList body;
    For(const Variable&      var,
        const Expression&    initial,
        const Expression&    condition,
        const Expression&    increment,
        const StatementList& body = {});
    std::string render() const;
};

class While
{
public:
    Expression    condition;
    StatementList body;
    While(const Expression& condition, const StatementList& body = {});
    std::string render() const;
};

class If
{
public:
    Expression    condition;
    StatementList body;
    If(const Expression& condition, const StatementList& body);
    std::string render() const;
};

class ElseIf
{
public:
    Expression    condition;
    StatementList body;
    ElseIf(const Expression& condition, const StatementList& body);
    std::string render() const;
};

class Else
{
public:
    StatementList body;
    explicit Else(const StatementList& body);
    std::string render() const;
};

class StoreGlobal
{
public:
    StoreGlobal(const Expression& ptr, const Expression& index, const Expression& value)
        : ptr{ptr}
        , index{index}
        , value{value}
    {
    }
    std::string render() const
    {
        return "store_cb(" + vrender(ptr) + "," + vrender(index) + ","
               + vrender(scale_factor ? (value * scale_factor.value()) : value)
               + ", store_cb_data, nullptr);";
    }

    Expression                ptr;
    Expression                index;
    Expression                value;
    std::optional<Expression> scale_factor;
};

// Planar version of StoreGlobal, so we remember the scale factor
// after a conversion to planar kernel
class StoreGlobalPlanar
{
public:
    StoreGlobalPlanar(const Variable&                  realPtr,
                      const Variable&                  imagPtr,
                      const Expression&                index,
                      const Variable&                  value,
                      const std::optional<Expression>& scale_factor)
        : realPtr{realPtr}
        , imagPtr{imagPtr}
        , index{index}
        , value{value}
        , scale_factor{scale_factor}
    {
    }
    std::string render() const
    {
        // Output two assignments
        return Assign{realPtr[index],
                      scale_factor ? Expression{value.x * scale_factor.value()}
                                   : Expression{value.x}}
                   .render()
               + Assign{imagPtr[index],
                        scale_factor ? Expression{value.y * scale_factor.value()}
                                     : Expression{value.y}}
                     .render();
    }

    Variable                  realPtr;
    Variable                  imagPtr;
    Expression                index;
    Variable                  value;
    std::optional<Expression> scale_factor;
};

class Butterfly
{
public:
    static const unsigned int precedence = 0;
    Butterfly(bool forward, const std::vector<Expression>& args)
        : forward(forward)
        , args(args)
    {
    }
    bool                    forward;
    std::vector<Expression> args;
    std::string             render() const;
};

class IntrinsicStore
{
public:
    IntrinsicStore(const Expression& ptr,
                   const Expression& voffset,
                   const Expression& soffset,
                   const Expression& value,
                   const Expression& rw_flag)
        : ptr{ptr}
        , voffset{voffset}
        , soffset{soffset}
        , value{value}
        , rw_flag{rw_flag}
    {
    }
    std::string render() const
    {
        return "store_intrinsic(" + vrender(ptr) + "," + vrender(voffset) + "," + vrender(soffset)
               + "," + vrender(scale_factor ? (value * scale_factor.value()) : value) + ","
               + vrender(rw_flag) + ");";
    }

    Expression                ptr;
    Expression                voffset;
    Expression                soffset;
    Expression                value;
    Expression                rw_flag;
    std::optional<Expression> scale_factor;
};

class IntrinsicLoadToDest
{
public:
    IntrinsicLoadToDest(const Expression& dest,
                        const Expression& data,
                        const Expression& voffset,
                        const Expression& soffset,
                        const Expression& rw_flag)
        : dest{dest}
        , data{data}
        , voffset{voffset}
        , soffset{soffset}
        , rw_flag{rw_flag}
    {
    }
    std::string render() const
    {
        return "intrinsic_load_to_dest(" + vrender(dest) + "," + vrender(data) + ","
               + vrender(voffset) + "," + vrender(soffset) + "," + vrender(rw_flag) + ");";
    }

    Expression dest;
    Expression data;
    Expression voffset;
    Expression soffset;
    Expression rw_flag;
};

// end of Statement class declarations

std::string Butterfly::render() const
{
    std::string func;
    if(forward)
    {
        func += "FwdRad" + std::to_string(args.size()) + "B1";
    }
    else
    {
        func += "InvRad" + std::to_string(args.size()) + "B1";
    }
    return Call{func, args}.render();
}

StatementList::StatementList() {}
StatementList::StatementList(const std::initializer_list<Statement>& il)
    : statements(il){};
std::string StatementList::render() const
{
    std::string r;
    for(auto s : statements)
        r += vrender(s) + "\n";
    return r;
}

void operator+=(StatementList& stmts, const Statement& s)
{
    stmts.statements.push_back(s);
}

void operator+=(StatementList& stmts, const StatementList& s)
{
    //    stmts.statements.insert(stmts.statements.end(), s.statements.cbegin(), s.statements.cend());
    for(auto x : s.statements)
    {
        stmts += x;
    }
}

For::For(const Variable&      var,
         const Expression&    initial,
         const Expression&    condition,
         const Expression&    increment,
         const StatementList& body)
    : var(var)
    , initial(initial)
    , condition(condition)
    , increment(increment)
    , body(body){};

std::string For::render() const
{
    std::string s;
    s += "for(";
    s += var.type + " " + var.name + " = ";
    s += vrender(initial) + "; ";
    s += vrender(condition) + "; ";

    // ++ and -- are nicer to read, so render those as a special case
    if(std::holds_alternative<Literal>(increment) && std::get<Literal>(increment).value == "1")
        s += "++" + var.name;
    else if(std::holds_alternative<Literal>(increment)
            && std::get<Literal>(increment).value == "-1")
        s += "--" + var.name;
    else
        s += var.name + " += " + vrender(increment);
    s += ") {\n ";
    s += body.render();
    s += "\n}";
    return s;
}

While::While(const Expression& condition, const StatementList& body)
    : condition(condition)
    , body(body){};
std::string While::render() const
{
    std::string s;
    s += "while(";
    s += vrender(condition) + ") {\n";
    s += body.render();
    s += "\n}";
    return s;
}

If::If(const Expression& condition, const StatementList& body)
    : condition(condition)
    , body(body){};
std::string If::render() const
{
    std::string s;
    s += "if(";
    s += vrender(condition);
    s += ") {\n";
    s += body.render();
    s += "\n}\n";
    return s;
}

ElseIf::ElseIf(const Expression& condition, const StatementList& body)
    : condition(condition)
    , body(body){};
std::string ElseIf::render() const
{
    std::string s;
    s += "else if(";
    s += vrender(condition);
    s += ") {\n";
    s += body.render();
    s += "\n}\n";
    return s;
}

Else::Else(const StatementList& body)
    : body(body){};
std::string Else::render() const
{
    std::string s;
    s += "else {\n";
    s += body.render();
    s += "\n}\n";
    return s;
}

//
// Functions
//

class Function
{
public:
    std::string   name;
    StatementList body;
    ArgumentList  arguments;
    TemplateList  templates;
    std::string   qualifier;
    unsigned int  launch_bounds = 0;

    explicit Function(const std::string& name)
        : name(name){};

    std::string render() const;
};

std::string Function::render() const
{
    std::string f;
    if(templates)
    {
        f += "template<" + templates.render_decl() + ">";
    }
    f += qualifier + " ";
    if(launch_bounds)
        f += "__launch_bounds__(" + std::to_string(launch_bounds) + ") ";
    f += "void " + name;
    f += "(" + arguments.render_decl() + ") {\n";
    f += body.render();
    f += "}\n";
    return f;
}

//
// Re-write helpers
//

// Base visitor class that actual visitor implementations can inherit
// from.
struct BaseVisitor
{
    BaseVisitor() = default;

    // Create operator() for each concrete type, so std::visit on a
    // variant will work.  "Statement" types all return a
    // StatementList.  Other types mostly return Expressions.  Each
    // method dispatches to a virtual visit_* method so we can
    // subclass just what we want.
#define MAKE_VISITOR_OPERATOR(RET, CLS) \
    RET operator()(const CLS& x)        \
    {                                   \
        return visit_##CLS(x);          \
    }

    MAKE_VISITOR_OPERATOR(Expression, ScalarVariable);
    MAKE_VISITOR_OPERATOR(Expression, Variable);
    MAKE_VISITOR_OPERATOR(Expression, Literal);
    MAKE_VISITOR_OPERATOR(Expression, ComplexLiteral);
    MAKE_VISITOR_OPERATOR(Expression, Add);
    MAKE_VISITOR_OPERATOR(Expression, Subtract);
    MAKE_VISITOR_OPERATOR(Expression, Multiply);
    MAKE_VISITOR_OPERATOR(Expression, Divide);
    MAKE_VISITOR_OPERATOR(Expression, Modulus);
    MAKE_VISITOR_OPERATOR(Expression, ShiftLeft);
    MAKE_VISITOR_OPERATOR(Expression, ShiftRight);
    MAKE_VISITOR_OPERATOR(Expression, And);
    MAKE_VISITOR_OPERATOR(Expression, BitAnd);
    MAKE_VISITOR_OPERATOR(Expression, Or);
    MAKE_VISITOR_OPERATOR(Expression, Less);
    MAKE_VISITOR_OPERATOR(Expression, LessEqual);
    MAKE_VISITOR_OPERATOR(Expression, Greater);
    MAKE_VISITOR_OPERATOR(Expression, GreaterEqual);
    MAKE_VISITOR_OPERATOR(Expression, Equal);
    MAKE_VISITOR_OPERATOR(Expression, NotEqual);
    MAKE_VISITOR_OPERATOR(Expression, UnaryMinus);
    MAKE_VISITOR_OPERATOR(Expression, Not);
    MAKE_VISITOR_OPERATOR(Expression, PreIncrement);
    MAKE_VISITOR_OPERATOR(Expression, PreDecrement);
    MAKE_VISITOR_OPERATOR(Expression, Ternary);
    MAKE_VISITOR_OPERATOR(Expression, LoadGlobal);
    MAKE_VISITOR_OPERATOR(Expression, ComplexMultiply);
    MAKE_VISITOR_OPERATOR(Expression, TwiddleMultiply);
    MAKE_VISITOR_OPERATOR(Expression, TwiddleMultiplyConjugate);
    MAKE_VISITOR_OPERATOR(Expression, Parens);
    MAKE_VISITOR_OPERATOR(Expression, CallExpr);
    MAKE_VISITOR_OPERATOR(Expression, IntrinsicLoad);

    MAKE_VISITOR_OPERATOR(StatementList, Assign);
    MAKE_VISITOR_OPERATOR(StatementList, Call);
    MAKE_VISITOR_OPERATOR(StatementList, CallbackDeclaration);
    MAKE_VISITOR_OPERATOR(StatementList, CommentLines);
    MAKE_VISITOR_OPERATOR(StatementList, Declaration);
    MAKE_VISITOR_OPERATOR(StatementList, LDSDeclaration);
    MAKE_VISITOR_OPERATOR(StatementList, For);
    MAKE_VISITOR_OPERATOR(StatementList, While);
    MAKE_VISITOR_OPERATOR(StatementList, If);
    MAKE_VISITOR_OPERATOR(StatementList, ElseIf);
    MAKE_VISITOR_OPERATOR(StatementList, Else);
    MAKE_VISITOR_OPERATOR(StatementList, StoreGlobal);
    MAKE_VISITOR_OPERATOR(StatementList, StoreGlobalPlanar);
    MAKE_VISITOR_OPERATOR(StatementList, LineBreak);
    MAKE_VISITOR_OPERATOR(StatementList, Return);
    MAKE_VISITOR_OPERATOR(StatementList, Break);
    MAKE_VISITOR_OPERATOR(StatementList, SyncThreads);
    MAKE_VISITOR_OPERATOR(StatementList, Butterfly);
    MAKE_VISITOR_OPERATOR(StatementList, IntrinsicStore);
    MAKE_VISITOR_OPERATOR(StatementList, IntrinsicLoadToDest);

    MAKE_VISITOR_OPERATOR(ArgumentList, ArgumentList);

    MAKE_VISITOR_OPERATOR(Function, Function);

    // operator for StatementList itself is a bit special - need to
    // visit each statement in the list
    StatementList operator()(const StatementList& x)
    {
        StatementList ret;
        for(const auto& stmt : x.statements)
        {
            StatementList new_stmts = std::visit(*this, stmt);
            std::copy(new_stmts.statements.begin(),
                      new_stmts.statements.end(),
                      std::back_inserter(ret.statements));
        }
        return ret;
    }

    // "visit" methods know how to visit their children.
    //
    // - TRIVIAL visitors are for types that have no children.
    //
    // - EXPR visitors are for Expression types whose only children
    //   are in a vector<Expression> named "exprs".
    //
    // - STATEMENT types return a StatementList.
    //
    // - Types that have children but don't fit the EXPR mold need
    //   their own hand-written visit function.
#define MAKE_TRIVIAL_VISIT(RET, CLS)      \
    virtual RET visit_##CLS(const CLS& x) \
    {                                     \
        return x;                         \
    }

#define MAKE_TRIVIAL_STATEMENT_VISIT(CLS)           \
    virtual StatementList visit_##CLS(const CLS& x) \
    {                                               \
        StatementList stmts;                        \
        stmts += Statement{x};                      \
        return stmts;                               \
    }

#define MAKE_EXPR_VISIT(CLS)                        \
    virtual Expression visit_##CLS(const CLS& x)    \
    {                                               \
        std::vector<Expression> args;               \
        for(const auto& arg : x.args)               \
            args.push_back(std::visit(*this, arg)); \
        return CLS{args};                           \
    }

    MAKE_EXPR_VISIT(Add);
    MAKE_EXPR_VISIT(And);
    MAKE_EXPR_VISIT(BitAnd);
    MAKE_EXPR_VISIT(Divide);
    MAKE_EXPR_VISIT(Equal);
    MAKE_EXPR_VISIT(Greater);
    MAKE_EXPR_VISIT(GreaterEqual);
    MAKE_EXPR_VISIT(Less);
    MAKE_EXPR_VISIT(LessEqual);
    MAKE_EXPR_VISIT(Modulus);
    MAKE_EXPR_VISIT(Multiply);
    MAKE_EXPR_VISIT(NotEqual);
    MAKE_EXPR_VISIT(Or);
    MAKE_EXPR_VISIT(ShiftLeft);
    MAKE_EXPR_VISIT(ShiftRight);
    MAKE_EXPR_VISIT(Subtract);

    MAKE_EXPR_VISIT(UnaryMinus);
    MAKE_EXPR_VISIT(Not);
    MAKE_EXPR_VISIT(PreIncrement);
    MAKE_EXPR_VISIT(PreDecrement);

    MAKE_EXPR_VISIT(LoadGlobal);

    MAKE_TRIVIAL_VISIT(Expression, ComplexMultiply);
    MAKE_TRIVIAL_VISIT(Expression, TwiddleMultiply);
    MAKE_TRIVIAL_VISIT(Expression, TwiddleMultiplyConjugate);

    MAKE_EXPR_VISIT(IntrinsicLoad);
    MAKE_EXPR_VISIT(Parens);

    MAKE_EXPR_VISIT(Ternary);
    MAKE_EXPR_VISIT(ComplexLiteral)

    MAKE_TRIVIAL_VISIT(Expression, ScalarVariable)
    MAKE_TRIVIAL_STATEMENT_VISIT(CallbackDeclaration)
    MAKE_TRIVIAL_STATEMENT_VISIT(LDSDeclaration)

    MAKE_TRIVIAL_VISIT(Expression, Literal)
    MAKE_TRIVIAL_STATEMENT_VISIT(CommentLines)
    MAKE_TRIVIAL_STATEMENT_VISIT(LineBreak)
    MAKE_TRIVIAL_STATEMENT_VISIT(Return)
    MAKE_TRIVIAL_STATEMENT_VISIT(Break)
    MAKE_TRIVIAL_STATEMENT_VISIT(SyncThreads)
    MAKE_TRIVIAL_STATEMENT_VISIT(Butterfly);
    MAKE_TRIVIAL_STATEMENT_VISIT(IntrinsicLoadToDest);

    MAKE_TRIVIAL_VISIT(Expression, Variable)

    virtual StatementList visit_StatementList(const StatementList& x)
    {
        auto y = StatementList();
        for(auto s : x.statements)
        {
            y += std::visit(*this, s);
        }
        return y;
    }

    virtual ArgumentList visit_ArgumentList(const ArgumentList& x)
    {
        auto y = ArgumentList();
        for(auto s : x.arguments)
        {
            y.append(std::get<Variable>(visit_Variable(s)));
        }
        return y;
    }

    virtual StatementList visit_Assign(const Assign& x)
    {
        auto lhs = std::get<Variable>(visit_Variable(x.lhs));
        auto rhs = std::visit(*this, x.rhs);
        return StatementList{Assign{lhs, rhs, x.oper}};
    }

    virtual Expression visit_CallExpr(const CallExpr& x)
    {
        auto y      = CallExpr(x);
        y.templates = visit_ArgumentList(x.templates);
        y.arguments.clear();
        y.arguments.reserve(x.arguments.size());
        for(const auto& arg : x.arguments)
            y.arguments.push_back(std::visit(*this, arg));
        return y;
    }

    virtual StatementList visit_Call(const Call& x)
    {
        auto y = std::get<CallExpr>(visit_CallExpr(x.expr));
        return StatementList{Call{y.name, y.templates, y.arguments}};
    }

    virtual StatementList visit_Declaration(const Declaration& x)
    {
        auto var = std::get<Variable>(visit_Variable(x.var));
        if(x.value)
        {
            return StatementList{Declaration(var, std::visit(*this, *x.value))};
        }
        return StatementList{Declaration(var)};
    }

    virtual StatementList visit_For(const For& x)
    {
        auto var       = std::get<Variable>(visit_Variable(x.var));
        auto initial   = std::visit(*this, x.initial);
        auto condition = std::visit(*this, x.condition);
        auto increment = std::visit(*this, x.increment);
        auto body      = visit_StatementList(x.body);
        return StatementList{For(var, initial, condition, increment, body)};
    }

    virtual StatementList visit_While(const While& x)
    {
        auto condition = std::visit(*this, x.condition);
        auto body      = visit_StatementList(x.body);
        return StatementList{While(condition, body)};
    }

    virtual StatementList visit_If(const If& x)
    {
        auto condition = std::visit(*this, x.condition);
        auto body      = visit_StatementList(x.body);
        return StatementList{If(condition, body)};
    }

    virtual StatementList visit_ElseIf(const ElseIf& x)
    {
        auto condition = std::visit(*this, x.condition);
        auto body      = visit_StatementList(x.body);
        return StatementList{ElseIf(condition, body)};
    }

    virtual StatementList visit_Else(const Else& x)
    {
        auto body = visit_StatementList(x.body);
        return StatementList{Else(body)};
    }

    virtual StatementList visit_StoreGlobal(const StoreGlobal& x)
    {
        auto ptr   = std::visit(*this, x.ptr);
        auto index = std::visit(*this, x.index);
        auto value = std::visit(*this, x.value);
        return StatementList{StoreGlobal(ptr, index, value)};
    }

    virtual StatementList visit_IntrinsicStore(const IntrinsicStore& x)
    {
        auto ptr     = std::visit(*this, x.ptr);
        auto voffset = std::visit(*this, x.voffset);
        auto soffset = std::visit(*this, x.soffset);
        auto value   = std::visit(*this, x.value);
        auto rw_flag = std::visit(*this, x.rw_flag);
        return StatementList{IntrinsicStore(ptr, voffset, soffset, value, rw_flag)};
    }

    virtual StatementList visit_StoreGlobalPlanar(const StoreGlobalPlanar& x)
    {
        auto                      realPtr = std::get<Variable>(visit_Variable(x.realPtr));
        auto                      imagPtr = std::get<Variable>(visit_Variable(x.imagPtr));
        auto                      index   = std::visit(*this, x.index);
        auto                      value   = std::get<Variable>(visit_Variable(x.value));
        std::optional<Expression> scale_factor;
        if(x.scale_factor)
            scale_factor = std::visit(*this, x.scale_factor.value());
        return StatementList{StoreGlobalPlanar(realPtr, imagPtr, index, value, scale_factor)};
    }

    virtual Function visit_Function(const Function& x)
    {
        auto y          = Function(x.name);
        y.body          = visit_StatementList(x.body);
        y.arguments     = visit_ArgumentList(x.arguments);
        y.templates     = visit_ArgumentList(x.templates);
        y.qualifier     = x.qualifier;
        y.launch_bounds = x.launch_bounds;
        return y;
    }
};

//
// Make planar
//

struct MakePlanarVisitor : public BaseVisitor
{
    std::string varname, rename, imname;

    MakePlanarVisitor(const std::string& varname)
        : varname(varname)
        , rename(varname + "re")
        , imname(varname + "im")
    {
    }

    ArgumentList visit_ArgumentList(const ArgumentList& x) override
    {
        ArgumentList y;
        for(auto a : x.arguments)
        {
            if(a.name == varname)
            {
                auto re = Variable(a);
                re.name = rename;
                re.type = "real_type_t<" + a.type + ">";
                auto im = Variable(a);
                im.name = imname;
                im.type = "real_type_t<" + a.type + ">";
                y.append(re);
                y.append(im);
            }
            else
            {
                y.append(a);
            }
        }
        return y;
    }

    StatementList visit_Assign(const Assign& x) override
    {
        StatementList stmts;
        if(x.lhs.name == varname && std::holds_alternative<Variable>(x.rhs))
        {
            // on lhs, lhs needs to be split; use .x and .y on rhs

            auto rhs = std::get<Variable>(x.rhs);

            auto re = Variable(x.lhs);
            re.name = rename;
            auto im = Variable(x.lhs);
            im.name = imname;

            // FIXME- Not every rhs is complex
            // stmts += Assign(re, rhs.x, x.oper);
            // stmts += Assign(im, rhs.y, x.oper);
            stmts += Assign(re, rhs, x.oper);
            stmts += Assign(im, rhs, x.oper);
            return stmts;
        }
        else if(std::holds_alternative<Variable>(x.rhs)
                && std::get<Variable>(x.rhs).name == varname)
        {
            // on rhs, rhs needs to be joined as a complex literal

            auto rhs = std::get<Variable>(x.rhs);
            auto re  = Variable(rhs);
            re.name  = rename;
            auto im  = Variable(rhs);
            im.name  = imname;
            stmts += Assign{x.lhs, ComplexLiteral{re, im}, x.oper};
            return stmts;
        }
        // callbacks don't support planar, so loads are just direct
        // memory accesses
        else if(std::holds_alternative<LoadGlobal>(x.rhs))
        {
            auto load = std::get<LoadGlobal>(x.rhs);
            auto ptr  = std::get<Variable>(load.args[0]);
            if(ptr.name == varname)
            {
                auto& idx = load.args[1];

                auto re = ptr;
                re.name = rename;
                auto im = ptr;
                im.name = imname;

                stmts += Assign{x.lhs, ComplexLiteral{re[idx], im[idx]}, x.oper};
                return stmts;
            }
        }
        // callbacks don't support planar
        else if(std::holds_alternative<IntrinsicLoad>(x.rhs))
        {
            auto load = std::get<IntrinsicLoad>(x.rhs);
            auto ptr  = std::get<Variable>(load.args[0]);
            if(ptr.name == varname)
            {
                auto& voffset = load.args[1];
                auto& soffset = load.args[2];
                auto& rw_flag = load.args[3];

                auto re = ptr;
                re.name = rename;
                auto im = ptr;
                im.name = imname;

                stmts += Assign{x.lhs,
                                ComplexLiteral{IntrinsicLoad({re, voffset, soffset, rw_flag}),
                                               IntrinsicLoad({im, voffset, soffset, rw_flag})},
                                x.oper};
                return stmts;
            }
        }

        return StatementList{x};
    }

    StatementList visit_StoreGlobal(const StoreGlobal& x) override
    {
        // callbacks don't support planar, so stores are just direct
        // memory accesses
        auto var = std::get<Variable>(x.ptr);

        if(var.name == varname)
        {
            auto re = var;
            re.name = rename;
            auto im = var;
            im.name = imname;

            auto value = std::get<Variable>(x.value);
            return {StoreGlobalPlanar{re, im, x.index, value, x.scale_factor}};
        }
        return StatementList{x};
    }

    StatementList visit_IntrinsicStore(const IntrinsicStore& x) override
    {
        // callbacks don't support planar
        auto var = std::get<Variable>(x.ptr);

        if(var.name == varname)
        {
            auto re = var;
            re.name = rename;
            auto im = var;
            im.name = imname;

            StatementList stmts;
            stmts += Call{
                "store_intrinsic",
                {re,
                 x.voffset,
                 x.soffset,
                 Literal{"real_type_t<scalar_type>("
                         + vrender(x.scale_factor ? (x.scale_factor.value() * x.value) : x.value)
                         + ".x)"},
                 x.rw_flag}};
            stmts += Call{
                "store_intrinsic",
                {im,
                 x.voffset,
                 x.soffset,
                 Literal{"real_type_t<scalar_type>("
                         + vrender(x.scale_factor ? (x.scale_factor.value() * x.value) : x.value)
                         + ".y)"},
                 x.rw_flag}};
            return stmts;
        }
        return StatementList{x};
    }
};

Function make_planar(const Function& f, const std::string& varname)
{
    auto visitor = MakePlanarVisitor(varname);
    return visitor(f);
}

//
// Make out of place
//
struct MakeOutOfPlaceVisitor : public BaseVisitor
{
    const std::vector<std::string> op_names;
    MakeOutOfPlaceVisitor(std::vector<std::string>&& op_names)
        : op_names(op_names)
    {
    }

    bool op_name_match(const std::string& s)
    {
        return std::find(op_names.begin(), op_names.end(), s) != op_names.end();
    }

    enum class ExpressionVisitMode
    {
        INPUT,
        OUTPUT,
    };
    ExpressionVisitMode mode = ExpressionVisitMode::INPUT;

    Expression visit_LoadGlobal(const LoadGlobal& x) override
    {
        mode = ExpressionVisitMode::INPUT;
        std::vector<Expression> args;
        for(const auto& arg : x.args)
            args.push_back(std::visit(*this, arg));
        return LoadGlobal{args};
    }
    Expression visit_IntrinsicLoad(const IntrinsicLoad& x) override
    {
        mode = ExpressionVisitMode::INPUT;
        std::vector<Expression> args;
        for(const auto& arg : x.args)
            args.push_back(std::visit(*this, arg));
        return IntrinsicLoad{args};
    }
    StatementList visit_StoreGlobal(const StoreGlobal& x) override
    {
        StatementList stmts;
        mode       = ExpressionVisitMode::OUTPUT;
        auto ptr   = std::visit(*this, x.ptr);
        auto index = std::visit(*this, x.index);
        auto value = std::visit(*this, x.value);
        stmts += StoreGlobal{ptr, index, value};
        return stmts;
    }
    StatementList visit_IntrinsicStore(const IntrinsicStore& x) override
    {
        StatementList stmts;
        mode         = ExpressionVisitMode::OUTPUT;
        auto ptr     = std::visit(*this, x.ptr);
        auto voffset = std::visit(*this, x.voffset);
        auto soffset = std::visit(*this, x.soffset);
        auto value   = std::visit(*this, x.value);
        auto rw_flag = std::visit(*this, x.rw_flag);
        stmts += IntrinsicStore{ptr, voffset, soffset, value, rw_flag};
        return stmts;
    }

    StatementList visit_Assign(const Assign& x) override
    {
        if(!op_name_match(x.lhs.name))
            return BaseVisitor::visit_Assign(x);
        mode         = ExpressionVisitMode::INPUT;
        auto in_lhs  = std::get<Variable>(visit_Variable(x.lhs));
        auto in_rhs  = std::visit(*this, x.rhs);
        mode         = ExpressionVisitMode::OUTPUT;
        auto out_lhs = std::get<Variable>(visit_Variable(x.lhs));
        auto out_rhs = std::visit(*this, x.rhs);

        StatementList ret;
        ret += Assign{in_lhs, in_rhs, x.oper};
        ret += Assign{out_lhs, out_rhs, x.oper};
        return ret;
    }

    ArgumentList visit_ArgumentList(const ArgumentList& x) override
    {
        ArgumentList ret;
        for(const auto& arg : x.arguments)
        {
            if(op_name_match(arg.name))
            {
                mode = ExpressionVisitMode::INPUT;
                ret.append(std::get<Variable>(visit_Variable(arg)));
                mode = ExpressionVisitMode::OUTPUT;
                ret.append(std::get<Variable>(visit_Variable(arg)));
            }
            else
                ret.append(arg);
        }
        return ret;
    }

    Expression visit_Variable(const Variable& x) override
    {
        if(!op_name_match(x.name))
            return x;

        Variable y{x};
        y.name += mode == ExpressionVisitMode::INPUT ? "_in" : "_out";
        return y;
    }

    StatementList visit_Declaration(const Declaration& x) override
    {
        if(!op_name_match(x.var.name))
            return BaseVisitor::visit_Declaration(x);

        StatementList ret;
        mode        = ExpressionVisitMode::INPUT;
        auto in_var = std::get<Variable>(visit_Variable(x.var));
        if(x.value)
            ret += Declaration{in_var, std::visit(*this, *x.value)};
        else
            ret += Declaration{in_var};
        mode         = ExpressionVisitMode::OUTPUT;
        auto out_var = std::get<Variable>(visit_Variable(x.var));
        if(x.value)
            ret += Declaration{out_var, std::visit(*this, *x.value)};
        else
            ret += Declaration{out_var};
        return ret;
    }

    Function visit_Function(const Function& x) override
    {
        Function y{x};
        y.name = "op_" + y.name;
        return BaseVisitor::visit_Function(y);
    }
};

Function make_outofplace(const Function& f)
{
    auto visitor = MakeOutOfPlaceVisitor({"buf", "stride", "stride0", "offset"});
    return visitor(f);
}

//
// Make in-place
//
struct MakeInPlaceVisitor : public BaseVisitor
{
    MakeInPlaceVisitor() = default;

    Function visit_Function(const Function& x) override
    {
        Function y{x};
        y.name = "ip_" + y.name;
        return BaseVisitor::visit_Function(y);
    }
};

Function make_inplace(const Function& f)
{
    auto visitor = MakeInPlaceVisitor();
    return visitor(f);
}

//
// Make inverse
//
static const char*  FORWARD_PREFIX     = "forward_";
static const size_t FORWARD_PREFIX_LEN = strlen(FORWARD_PREFIX);
static const char*  INVERSE_PREFIX     = "inverse_";

struct MakeInverseVisitor : public BaseVisitor
{
    Expression visit_TwiddleMultiply(const TwiddleMultiply& x) override
    {
        return TwiddleMultiplyConjugate{x.a, x.b};
    }

    StatementList visit_Butterfly(const Butterfly& x) override
    {
        return {Butterfly{false, x.args}};
    }

    Expression visit_CallExpr(const CallExpr& x) override
    {
        auto pos = x.name.rfind(FORWARD_PREFIX, 0);
        if(pos == 0)
        {
            CallExpr y{x};
            y.name.replace(0, FORWARD_PREFIX_LEN, INVERSE_PREFIX);
            return y;
        }
        return BaseVisitor::visit_CallExpr(x);
    }

    Function visit_Function(const Function& x) override
    {
        auto pos = x.name.rfind(FORWARD_PREFIX, 0);
        if(pos != 0)
            return x;
        Function y{x};
        y.name.replace(0, FORWARD_PREFIX_LEN, INVERSE_PREFIX);
        return BaseVisitor::visit_Function(y);
    }
};

Function make_inverse(const Function& f)
{
    auto visitor = MakeInverseVisitor();
    return visitor(f);
}

//
// Make runtime-compileable
//

struct MakeRTCVisitor : public BaseVisitor
{
    MakeRTCVisitor(const std::string& kernel_name, bool enable_scaling)
        : kernel_name(kernel_name)
        , enable_scaling(enable_scaling)
        , scale_factor("scale_factor", "const real_type_t<scalar_type>")
    {
    }
    Function visit_Function(const Function& x) override
    {
        if(x.qualifier != "__global__")
            return x;
        // give function C linkage so caller doesn't have to do C++ name
        // mangling
        Function y{x};
        y.qualifier = "extern \"C\" __global__";
        // rocfft constructed a name for the function
        y.name = kernel_name;
        // assume some global-scope typedefs + consts have removed
        // the need for template args.
        y.templates.arguments.clear();

        // scaling requires an extra argument with the scale factor
        if(enable_scaling)
            y.arguments.append(scale_factor);

        return BaseVisitor::visit_Function(y);
    }

    StatementList visit_StoreGlobal(const StoreGlobal& x) override
    {
        StoreGlobal y{x};
        // multiply by scale factor when storing to global, if requested
        if(enable_scaling)
            y.scale_factor = scale_factor;
        return StatementList{y};
    }

    StatementList visit_StoreGlobalPlanar(const StoreGlobalPlanar& x) override
    {
        StoreGlobalPlanar y{x};
        // multiply by scale factor when storing to global, if requested
        if(enable_scaling)
            y.scale_factor = scale_factor;
        return StatementList{y};
    }

    StatementList visit_IntrinsicStore(const IntrinsicStore& x) override
    {
        IntrinsicStore y{x};
        // multiply by scale factor when storing to global, if requested
        if(enable_scaling)
            y.scale_factor = scale_factor;
        return StatementList{y};
    }

    std::string kernel_name;
    bool        enable_scaling;
    Variable    scale_factor;
};

Function make_rtc(const Function& f, const std::string& kernel_name, bool enable_scaling)
{
    auto visitor = MakeRTCVisitor(kernel_name, enable_scaling);
    return visitor(f);
}
