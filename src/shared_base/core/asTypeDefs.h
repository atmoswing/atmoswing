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
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef AS_TYPE_DEFS_H
#define AS_TYPE_DEFS_H

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#include <Eigen/StdVector>

//---------------------------------
// Structures
//---------------------------------

// Coordinates
struct Coo {
    double x;  // East
    double y;  // North
};

// A time structure
struct Time {
    int year;
    int month;
    int day;
    int hour;
    int min;
    int sec;
};

//---------------------------------
// std vector
//---------------------------------

using std::vector;
using vb = vector<bool>;
using vs = vector<short>;
using vi = vector<int>;
using vl = vector<long>;
using vf = vector<float>;
using vd = vector<double>;
using vstds = vector<std::string>;
using vwxs = vector<wxString>;

using vvb = vector<vb>;
using vvi = vector<vi>;
using vvf = vector<vf>;
using vvd = vector<vd>;
using vvwxs = vector<vwxs>;

//---------------------------------
// Eigen3 arrays
//---------------------------------

// Arrays are used for element-wise calculations. It is often the case here.
using a1i = Eigen::Array<int, Eigen::Dynamic, 1>;
using a1f = Eigen::Array<float, Eigen::Dynamic, 1>;
using a1d = Eigen::Array<double, Eigen::Dynamic, 1>;
using a2f = Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using va1f = vector<a1f, Eigen::aligned_allocator<a1f>>;
using va1d = vector<a1d, Eigen::aligned_allocator<a1d>>;
using va2f = vector<a2f, Eigen::aligned_allocator<a2f>>;
using vpa2f = vector<a2f*, Eigen::aligned_allocator<a2f*>>;
using vva2f = vector<vector<a2f, Eigen::aligned_allocator<a2f>>>;
using vvva2f = vector<vector<vector<a2f, Eigen::aligned_allocator<a2f>>>>;

#endif
