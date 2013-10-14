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

#include "include_tests.h"
#include "asGeoArea.h"

#include "UnitTest++.h"

namespace
{

TEST(ConstructorLimitsException)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = -10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = -10;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;

    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asGeoArea geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR), asException);
    }
}

TEST(ConstructorAlternativeLimitsException)
{
    double Umin = -10;
    double Uwidth = 30;
    double Vmin = 30;
    double Vwidth = 10;

    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asGeoArea geoarea(WGS84, Umin, Uwidth, Vmin, Vwidth), asException);
    }
}

TEST(CheckConsistency)
{
    double Umin = 10;
    double Uwidth = 10;
    double Vmin = 30;
    double Vwidth = 10;
    asGeoArea geoarea(WGS84, Umin, Uwidth, Vmin, Vwidth);

    CHECK_CLOSE(30, geoarea.GetCornerLL().v, 0.01);
    CHECK_CLOSE(30, geoarea.GetCornerLR().v, 0.01);
    CHECK_CLOSE(40, geoarea.GetCornerUL().v, 0.01);
    CHECK_CLOSE(40, geoarea.GetCornerUR().v, 0.01);
}

TEST(IsRectangleTrue)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = 10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = 10;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;
    asGeoArea geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR);

    CHECK_EQUAL(true, geoarea.IsRectangle());
}

TEST(IsRectangleFalse)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = 10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = 15;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;
    asGeoArea geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR);

    CHECK_EQUAL(false, geoarea.IsRectangle());
}

TEST(GetBounds)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = 10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = 10;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;
    asGeoArea geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR);

    CHECK_CLOSE(10, geoarea.GetUmin(), 0.01);
    CHECK_CLOSE(30, geoarea.GetVmin(), 0.01);
    CHECK_CLOSE(10, geoarea.GetUwidth(), 0.01);
    CHECK_CLOSE(10, geoarea.GetVwidth(), 0.01);
}

TEST(GetCenter)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = 10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = 10;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;
    asGeoArea geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR);

    Coo center = geoarea.GetCenter();
    CHECK_CLOSE(15, center.u, 0.01);
    CHECK_CLOSE(35, center.v, 0.01);
}

TEST(NegativeSize)
{
    double Umin = 10;
    double Uwidth = -7;
    double Vmin = 46;
    double Vwidth = -2;

    asGeoArea geoarea(WGS84, Umin, Uwidth, Vmin, Vwidth, asNONE, asNONE, asFLAT_ALLOWED);

    CHECK_CLOSE(10, geoarea.GetUmin(), 0.01);
    CHECK_CLOSE(46, geoarea.GetVmin(), 0.01);
    CHECK_CLOSE(0, geoarea.GetUwidth(), 0.01);
    CHECK_CLOSE(0, geoarea.GetVwidth(), 0.01);
}

TEST(ProjConvert)
{
    double Umin = 7;
    double Uwidth = 3;
    double Vmin = 46;
    double Vwidth = 2;

    asGeoArea geoarea(WGS84, Umin, Uwidth, Vmin, Vwidth);

    geoarea.ProjConvert(CH1903);

    CHECK_CLOSE(566017.05, geoarea.GetCornerLL().u, 2);
    CHECK_CLOSE(94366.97, geoarea.GetCornerLL().v, 2);
    CHECK_CLOSE(798403.40, geoarea.GetCornerLR().u, 2);
    CHECK_CLOSE(97511.91, geoarea.GetCornerLR().v, 2);
    CHECK_CLOSE(791142.61, geoarea.GetCornerUR().u, 2);
    CHECK_CLOSE(319746.83, geoarea.GetCornerUR().v, 2);
    CHECK_CLOSE(567262.39, geoarea.GetCornerUL().u, 2);
    CHECK_CLOSE(316716.97, geoarea.GetCornerUL().v, 2);
}

}
