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
	wxPrintf("Testing geo area management...\n");
	
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = -10;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;

    if(g_unitTestExceptions)
    {
        CHECK_THROW(asGeoArea geoarea(CornerUL, CornerUR, CornerLL, CornerLR), asException);
    }
}

TEST(ConstructorAlternativeLimitsException)
{
    if(g_unitTestExceptions)
    {
        double Xmin = -10;
        double Xwidth = 30;
        double Ymin = 30;
        double Ywidth = 10;
        CHECK_THROW(asGeoArea geoarea(Xmin, Xwidth, Ymin, Ywidth), asException);
    }
}

TEST(CheckConsistency)
{
    double Xmin = 10;
    double Xwidth = 10;
    double Ymin = 30;
    double Ywidth = 10;
    asGeoArea geoarea(Xmin, Xwidth, Ymin, Ywidth);

    CHECK_CLOSE(30, geoarea.GetCornerLL().y, 0.01);
    CHECK_CLOSE(30, geoarea.GetCornerLR().y, 0.01);
    CHECK_CLOSE(40, geoarea.GetCornerUL().y, 0.01);
    CHECK_CLOSE(40, geoarea.GetCornerUR().y, 0.01);
}

TEST(IsRectangleTrue)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = 10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = 10;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;
    asGeoArea geoarea(CornerUL, CornerUR, CornerLL, CornerLR);

    CHECK_EQUAL(true, geoarea.IsRectangle());
}

TEST(IsRectangleFalse)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = 10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = 15;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;
    asGeoArea geoarea(CornerUL, CornerUR, CornerLL, CornerLR);

    CHECK_EQUAL(false, geoarea.IsRectangle());
}

TEST(GetBounds)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = 10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = 10;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;
    asGeoArea geoarea(CornerUL, CornerUR, CornerLL, CornerLR);

    CHECK_CLOSE(10, geoarea.GetXmin(), 0.01);
    CHECK_CLOSE(30, geoarea.GetYmin(), 0.01);
    CHECK_CLOSE(10, geoarea.GetXwidth(), 0.01);
    CHECK_CLOSE(10, geoarea.GetYwidth(), 0.01);
}

TEST(GetCenter)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = 10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = 10;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;
    asGeoArea geoarea(CornerUL, CornerUR, CornerLL, CornerLR);

    Coo center = geoarea.GetCenter();
    CHECK_CLOSE(15, center.x, 0.01);
    CHECK_CLOSE(35, center.y, 0.01);
}

TEST(NegativeSize)
{
    double Xmin = 10;
    double Xwidth = -7;
    double Ymin = 46;
    double Ywidth = -2;

    asGeoArea geoarea(Xmin, Xwidth, Ymin, Ywidth, asNONE, asNONE, asFLAT_ALLOWED);

    CHECK_CLOSE(10, geoarea.GetXmin(), 0.01);
    CHECK_CLOSE(46, geoarea.GetYmin(), 0.01);
    CHECK_CLOSE(0, geoarea.GetXwidth(), 0.01);
    CHECK_CLOSE(0, geoarea.GetYwidth(), 0.01);
}

}
