/////////////////////////////////////////////////////////////////////////////
// Name:        matrix2d.h
// Author:      John Labenski
// Created:     07/01/02
// Copyright:   John Labenski, 2002
// License:     wxWidgets v2
/////////////////////////////////////////////////////////////////////////////

#ifndef __WXTHINGS_MATRIX2D_H__
#define __WXTHINGS_MATRIX2D_H__

#include "wx/things/thingdef.h"
#include "wx/object.h"
#include "wx/arrstr.h"
#include "wx/gdicmn.h" // for wxSize

class WXDLLIMPEXP_FWD_CORE wxArrayInt;

//----------------------------------------------------------------------------
//  wxThingsMatrix2D - a generic double valued 2D ref counted matrix class
//----------------------------------------------------------------------------
class wxThingsMatrix2D
        : public wxObject
{
public:
    wxThingsMatrix2D()
    {
    }

    wxThingsMatrix2D(const wxThingsMatrix2D &matrix, bool full_copy = false)
    {
        Create(matrix, full_copy);
    }

    wxThingsMatrix2D(int width, int height, bool zero = true)
    {
        Create(width, height, zero);
    }

    wxThingsMatrix2D(int width, int height, const double *data)
    {
        Create(width, height, data);
    }

    wxThingsMatrix2D(int width, int height, double *data, bool static_data)
    {
        Create(width, height, data, static_data);
    }

    virtual ~wxThingsMatrix2D();

    // Create a wxThingsMatrix2D by Ref(ing) or doing a full unrefed copy after first
    //  unrefing the internal data
    bool Create(const wxThingsMatrix2D &matrix, bool full_copy = false);

    // Create an empty matrix and zero it if zero == true
    bool Create(int width, int height, bool zero = true);

    // Create a matrix by memcpy(ing) the input data
    bool Create(int width, int height, const double *matrix);

    // Create a matrix by using malloc(ed) data and free(ing) it when done if
    //   static_data == true. If static_data == false then the data must exist
    //   for the life of this class.
    bool Create(int width, int height, double *matrix, bool static_data);

    // Create the matrix as a square identity matrix with ones along the diagonal.
    bool CreateIdentity(int size);

    // Make a full copy, not just a refed one
    bool Copy(const wxThingsMatrix2D &matrix);

    // destroy the refed data, reducing the ref count by 1
    void Destroy();

    // Is this matrix created and has a valid width and height
    bool Ok() const;

    // Check if the matrix is exactly equal to this one using memcmp, either
    //  matrix may not be Ok() and be of different sizes, returning false.
    //  Note: the equality operator only checks if the ref data is the same.
    bool IsEqual(const wxThingsMatrix2D &matrix) const;

    // Get the size of the matrix, width = # cols, height = # rows
    int GetWidth() const;

    int GetHeight() const;

    wxSize GetSize() const;

    bool PointInMatrix(int x, int y) const;

    // Get/Set the value of a matrix element, zero based indexes
    double GetValue(int x, int y) const;

    void SetValue(int x, int y, double value);

    // Insert matrix at this row position, matricies must have same number of cols
    wxThingsMatrix2D InsertRows(int row, const wxThingsMatrix2D &matrix) const;

    // Insert matrix at this col position, matricies must have same number of rows
    wxThingsMatrix2D InsertCols(int col, const wxThingsMatrix2D &matrix) const;

    // Append new rows to the matrix, matricies must have same number of cols
    wxThingsMatrix2D AppendRows(const wxThingsMatrix2D &matrix) const
    {
        return InsertRows(-1, matrix);
    }

    // Append new cols to the matrix, matricies must have same number of rows
    wxThingsMatrix2D AppendCols(const wxThingsMatrix2D &matrix) const
    {
        return InsertCols(-1, matrix);
    }

    // Get a sub matrix inside of the this one, returns a new matrix
    wxThingsMatrix2D SubMatrix(const wxRect &rect) const;

    // Get a sub matrix of rows inside of the this one, returns a new matrix
    wxThingsMatrix2D SubRows(int start_row, int end_row) const;

    // Get a sub matrix of cols inside of the this one, returns a new matrix
    wxThingsMatrix2D SubCols(int start_col, int end_col) const;

    // Reshape the matrix by setting a new width and height. The data is not
    //  changed. Internally the data is stored as a linear array, such that
    //  element = x + y * width.
    bool Reshape(int width, int height);

    // Get the data array as a pointer of size width*height
    double *GetData() const;

    // Set the values for all elements to the given one
    bool Fill(double value);

    // Add this matrix to another matrix element by element, must have same dimensions
    wxThingsMatrix2D Add(const wxThingsMatrix2D &b) const;

    // Add this single value to all elements, returns a new matrix with the result
    wxThingsMatrix2D Add(double scalar) const;

    // Matrix multiplication, returns a new matrix with the result
    wxThingsMatrix2D Mult(const wxThingsMatrix2D &b) const;

    // Hadamard matrix multiplication, element-wise multiplication, returns a new matrix
    wxThingsMatrix2D MultElement(const wxThingsMatrix2D &b) const;

    // Multiply each element of the matrix by this value, returns a new matrix
    wxThingsMatrix2D Mult(double scalar) const;

    // Raise each element of the matrix to this power, returns a new matrix
    wxThingsMatrix2D Pow(double p) const;

    // Raise each element of the matrix to n^(element value), returns a new matrix
    wxThingsMatrix2D PowN(double n) const;

    // Take the natural log of each element of the matrix, returns a new matrix
    wxThingsMatrix2D Log() const;

    // Take the log base 10 of each element of the matrix, returns a new matrix
    wxThingsMatrix2D Log10() const;

    // Set each element to e^(element value), returns a new matrix
    wxThingsMatrix2D Exp() const;
    // Set each element to 10^(element value), returns a new matrix
    //    see PowN()
    //wxThingsMatrix2D Exp10() const;

    // Get the sum of all the elements in the matrix
    double Sum() const;

    // Get the trace of the matrix, the sum of the diagonal elements. The
    //   matrix should be square, but this just adds diagnals from upper left
    //   to lower right.
    double Trace() const;

    // Swap the rows and cols, returns a new matrix
    wxThingsMatrix2D Transpose() const;

    // Rotate the matrix by clockwise*(90 degrees), returns a new matrix
    //  clockwise can be negative to rotate counter-clockwise
    wxThingsMatrix2D Rotate90(int clockwise) const;

    // Mirror the matrix either horizontally or vertically, returns a new matrix
    wxThingsMatrix2D Mirror(bool horizontally = true) const;

    // rotate matrix by 45 deg, must be square and sides odd, returns a new matrix
    wxThingsMatrix2D RotateSquare45(bool clockwise) const;

    // normalize the sum of the values to this
    void Normalize(double sum = 1.0);

    // print to a string, separating cols and rows by given strings
    wxString ToString(const wxString &colSep = wxT(" "), const wxString &rowSep = wxT("\n")) const;

    // Load a file from disk
    bool LoadFile(const wxString &filename, const wxArrayInt *cols = NULL);

    wxArrayString m_file_comments;

    friend wxThingsMatrix2D operator+(wxThingsMatrix2D &a, wxThingsMatrix2D &b)
    {
        return a.Add(b);
    }

    friend wxThingsMatrix2D operator+(double scalar, wxThingsMatrix2D &b)
    {
        return b.Add(scalar);
    }

    friend wxThingsMatrix2D operator*(wxThingsMatrix2D &a, wxThingsMatrix2D &b)
    {
        return a.Mult(b);
    }

    friend wxThingsMatrix2D operator*(double scalar, wxThingsMatrix2D &a)
    {
        return a.Mult(scalar);
    }

    wxThingsMatrix2D &operator=(const wxThingsMatrix2D &matrix)
    {
        if ((*this) != matrix)
            wxObject::Ref(matrix);
        return *this;
    }

    bool operator==(const wxThingsMatrix2D &matrix)
    {
        if (!Ok() || !matrix.Ok())
            return false;
        return GetData() == matrix.GetData();
    }

    bool operator!=(const wxThingsMatrix2D &matrix)
    {
        return !(*this == matrix);
    }

private:
DECLARE_DYNAMIC_CLASS(wxThingsMatrix2D)
};

#endif // __WXTHINGS_MATRIX2D_H__
