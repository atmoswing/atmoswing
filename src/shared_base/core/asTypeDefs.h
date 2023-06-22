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
typedef struct Coo {
    double x;  // East
    double y;  // North
} Coo;

// A time structure
typedef struct Time {
    int year;
    int month;
    int day;
    int hour;
    int min;
    int sec;
} Time;

//---------------------------------
// std vector
//---------------------------------

using std::vector;
typedef vector<bool> vb;
typedef vector<short> vs;
typedef vector<int> vi;
typedef vector<long> vl;
typedef vector<float> vf;
typedef vector<double> vd;
typedef vector<std::string> vstds;
typedef vector<wxString> vwxs;

typedef vector<vb> vvb;
typedef vector<vi> vvi;
typedef vector<vf> vvf;
typedef vector<vd> vvd;
typedef vector<vwxs> vvwxs;

//---------------------------------
// Eigen3 arrays
//---------------------------------

// Arrays are used for element-wise calculations. It is often the case here.
typedef Eigen::Array<int, Eigen::Dynamic, 1> a1i;
typedef Eigen::Array<float, Eigen::Dynamic, 1> a1f;
typedef Eigen::Array<double, Eigen::Dynamic, 1> a1d;
typedef Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a2f;
typedef vector<a1f, Eigen::aligned_allocator<a1f>> va1f;
typedef vector<a1d, Eigen::aligned_allocator<a1d>> va1d;
typedef vector<a2f, Eigen::aligned_allocator<a2f>> va2f;
typedef vector<a2f*, Eigen::aligned_allocator<a2f*>> vpa2f;
typedef vector<vector<a2f, Eigen::aligned_allocator<a2f>>> vva2f;
typedef vector<vector<vector<a2f, Eigen::aligned_allocator<a2f>>>> vvva2f;

#endif
