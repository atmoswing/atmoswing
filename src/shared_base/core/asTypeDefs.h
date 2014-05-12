/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 * 
 * When distributing Covered Code, include this CDDL Header Notice in 
 * each file and include the License file (licence.txt). If applicable, 
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 * 
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#ifndef ASTYPEDEFS_H_
#define ASTYPEDEFS_H_

//---------------------------------
// Structures
//---------------------------------

// Coordinates
typedef struct
{
    double u; // East
    double v; // North
} Coo;

// A time structure
typedef struct
{
    int year;
    int month;
    int day;
    int hour;
    int min;
    int sec;
} TimeStruct;



//---------------------------------
// std vector
//---------------------------------

typedef std::vector < bool > VectorBool;
typedef std::vector < short > VectorShort;
typedef std::vector < int > VectorInt;
typedef std::vector < float > VectorFloat;
typedef std::vector < double > VectorDouble;
typedef std::vector < std::string > VectorStdString;
typedef std::vector < wxString > VectorString;

typedef std::vector < VectorBool > VVectorBool;
typedef std::vector < VectorShort > VVectorShort;
typedef std::vector < VectorInt > VVectorInt;
typedef std::vector < VectorFloat > VVectorFloat;
typedef std::vector < VectorDouble > VVectorDouble;
typedef std::vector < VectorStdString > VVectorStdString;
typedef std::vector < VectorString > VVectorString;



//---------------------------------
// Eigen3 matrices
//---------------------------------

// Matrices are used for real linear algebra.
typedef Eigen::Matrix < int , Eigen::Dynamic , Eigen::Dynamic , Eigen::RowMajor > Matrix2DInt;
typedef Eigen::Matrix < float , Eigen::Dynamic , Eigen::Dynamic , Eigen::RowMajor > Matrix2DFloat;
typedef Eigen::Matrix < double , Eigen::Dynamic , Eigen::Dynamic , Eigen::RowMajor > Matrix2DDouble;

// Arrays are used for element-wise calculations. It is often the case here.
typedef Eigen::Array < int , Eigen::Dynamic , 1 > Array1DInt;
typedef Eigen::Array < int , Eigen::Dynamic , Eigen::Dynamic , Eigen::RowMajor > Array2DInt;
typedef std::vector < Array1DInt > VArray1DInt;
typedef std::vector < Array2DInt > VArray2DInt;
typedef std::vector < Array2DInt* > VpArray2DInt;
typedef std::vector < std::vector < Array2DInt > > VVArray2DInt;
typedef std::vector < std::vector < Array2DInt* > > VVpArray2DInt;
typedef std::vector < std::vector < std::vector < Array2DInt > > > VVVArray2DInt;
typedef std::vector < std::vector < std::vector < Array2DInt* > > > VVVpArray2DInt;

typedef Eigen::Array < float , Eigen::Dynamic , 1 > Array1DFloat;
typedef Eigen::Array < float , Eigen::Dynamic , Eigen::Dynamic , Eigen::RowMajor > Array2DFloat;
typedef std::vector < Array1DFloat > VArray1DFloat;
typedef std::vector < Array2DFloat > VArray2DFloat;
typedef std::vector < Array2DFloat* > VpArray2DFloat;
typedef std::vector < std::vector < Array2DFloat > > VVArray2DFloat;
typedef std::vector < std::vector < Array2DFloat* > > VVpArray2DFloat;
typedef std::vector < std::vector < std::vector < Array2DFloat > > > VVVArray2DFloat;
typedef std::vector < std::vector < std::vector < Array2DFloat* > > > VVVpArray2DFloat;

typedef Eigen::Array < double , Eigen::Dynamic , 1  > Array1DDouble;
typedef Eigen::Array < double , Eigen::Dynamic , Eigen::Dynamic , Eigen::RowMajor > Array2DDouble;
typedef std::vector < Array1DDouble > VArray1DDouble;
typedef std::vector < Array2DDouble > VArray2DDouble;
typedef std::vector < Array2DDouble* > VpArray2DDouble;
typedef std::vector < std::vector < Array2DDouble > > VVArray2DDouble;
typedef std::vector < std::vector < Array2DDouble* > > VVpArray2DDouble;
typedef std::vector < std::vector < std::vector < Array2DDouble > > > VVVArray2DDouble;
typedef std::vector < std::vector < std::vector < Array2DDouble* > > > VVVpArray2DDouble;




//---------------------------------
// NaN & Inf
//---------------------------------

/* NaN (http://www.cplusplus.com/reference/limits/numeric_limits/) */
static const int NaNInt = numeric_limits<int>::max();
static const float NaNFloat = numeric_limits<float>::quiet_NaN();
static const double NaNDouble = numeric_limits<double>::quiet_NaN();

/* Inf (http://msdn.microsoft.com/en-us/library/6hthw3cb%28VS.80%29.aspx) */
const double InfFloat = numeric_limits<float>::infinity();
const double InfDouble = numeric_limits<double>::infinity();
const double InfLongDouble = numeric_limits<long double>::infinity();


#endif
