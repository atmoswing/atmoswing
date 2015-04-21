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
#include "asGeoAreaComposite.h"

#include "UnitTest++.h"

namespace
{

TEST(ConstructorOneArea)
{
	wxPrintf("Testing geo area composites...\n");
	
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = 10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = 10;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;
    asGeoAreaComposite geoarea(CornerUL, CornerUR, CornerLL, CornerLR);

    CHECK_EQUAL(1, geoarea.GetNbComposites());
}

TEST(ConstructorTwoAreas)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = -10;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;
    asGeoAreaComposite geoarea(CornerUL, CornerUR, CornerLL, CornerLR);

    CHECK_EQUAL(2, geoarea.GetNbComposites());
}

TEST(ConstructorAlternativeOneArea)
{
    double Xmin = 10;
    double Xwidth = 20;
    double Ymin = 30;
    double Ywidth = 40;
    asGeoAreaComposite geoarea(Xmin, Xwidth, Ymin, Ywidth);

    CHECK_EQUAL(1, geoarea.GetNbComposites());
}

TEST(ConstructorAlternativeTwoAreas)
{
    double Xmin = -10;
    double Xwidth = 20;
    double Ymin = 30;
    double Ywidth = 40;
    asGeoAreaComposite geoarea(Xmin, Xwidth, Ymin, Ywidth);

    CHECK_EQUAL(2, geoarea.GetNbComposites());
}

TEST(CheckConsistency)
{
    double Xmin = -5;
    double Xwidth = 20;
    double Ymin = 30;
    double Ywidth = 40;
    asGeoAreaComposite geoarea(Xmin, Xwidth, Ymin, Ywidth);

    CHECK_CLOSE(355, geoarea.GetCornerUL().x, 0.01);
    CHECK_CLOSE(355, geoarea.GetCornerLL().x, 0.01);
    CHECK_CLOSE(15, geoarea.GetCornerUR().x, 0.01);
    CHECK_CLOSE(15, geoarea.GetCornerLR().x, 0.01);
}

TEST(CheckConsistencyException)
{
    if(g_unitTestExceptions)
    {
        double Xmin = 10;
        double Xwidth = 10;
        double Ymin = 40;
        double Ywidth = 30;
        CHECK_THROW(asGeoAreaComposite geoarea(Xmin, Xwidth, Ymin, Ywidth), asException);
    }
}

TEST(IsSquareTrue)
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
    asGeoAreaComposite geoarea(CornerUL, CornerUR, CornerLL, CornerLR);

    CHECK_EQUAL(true, geoarea.IsRectangle());
}

TEST(IsSquareFalse)
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
    asGeoAreaComposite geoarea(CornerUL, CornerUR, CornerLL, CornerLR);

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
    asGeoAreaComposite geoarea(CornerUL, CornerUR, CornerLL, CornerLR);

    CHECK_CLOSE(10, geoarea.GetXmin(), 0.01);
    CHECK_CLOSE(30, geoarea.GetYmin(), 0.01);
    CHECK_CLOSE(20, geoarea.GetXmax(), 0.01);
    CHECK_CLOSE(40, geoarea.GetYmax(), 0.01);
}

TEST(GetBoundsSplitted)
{
    double Xmin = -10;
    double Xwidth = 20;
    double Ymin = 30;
    double Ywidth = 40;
    asGeoAreaComposite geoarea(Xmin, Xwidth, Ymin, Ywidth);

    CHECK_CLOSE(0, geoarea.GetXmin(), 0.01);
    CHECK_CLOSE(30, geoarea.GetYmin(), 0.01);
    CHECK_CLOSE(360, geoarea.GetXmax(), 0.01);
    CHECK_CLOSE(70, geoarea.GetYmax(), 0.01);
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
    asGeoAreaComposite geoarea(CornerUL, CornerUR, CornerLL, CornerLR);

    Coo center = geoarea.GetCenter();
    CHECK_CLOSE(15, center.x, 0.01);
    CHECK_CLOSE(35, center.y, 0.01);
}

TEST(GetCenterSplitted)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    asGeoAreaComposite geoarea(CornerUL, CornerUR, CornerLL, CornerLR);

    Coo center = geoarea.GetCenter();
    CHECK_CLOSE(345, center.x, 0.01);
    CHECK_CLOSE(35, center.y, 0.01);
}

TEST(GetCenterSplittedEdge)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -10;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -10;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    asGeoAreaComposite geoarea(CornerUL, CornerUR, CornerLL, CornerLR);

    Coo center = geoarea.GetCenter();
    CHECK_CLOSE(360, center.x, 0.01);
    CHECK_CLOSE(35, center.y, 0.01);
}

TEST(GetCornersSplitted)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    asGeoAreaComposite geoarea(CornerUL, CornerUR, CornerLL, CornerLR);

    CHECK_CLOSE(0, geoarea.GetComposite(0).GetCornerUL().x, 0.01);
    CHECK_CLOSE(40, geoarea.GetComposite(0).GetCornerUL().y, 0.01);
    CHECK_CLOSE(10, geoarea.GetComposite(0).GetCornerUR().x, 0.01);
    CHECK_CLOSE(40, geoarea.GetComposite(0).GetCornerUR().y, 0.01);
    CHECK_CLOSE(0, geoarea.GetComposite(0).GetCornerLL().x, 0.01);
    CHECK_CLOSE(30, geoarea.GetComposite(0).GetCornerLL().y, 0.01);
    CHECK_CLOSE(10, geoarea.GetComposite(0).GetCornerLR().x, 0.01);
    CHECK_CLOSE(30, geoarea.GetComposite(0).GetCornerLR().y, 0.01);

    CHECK_CLOSE(320, geoarea.GetComposite(1).GetCornerUL().x, 0.01);
    CHECK_CLOSE(40, geoarea.GetComposite(1).GetCornerUL().y, 0.01);
    CHECK_CLOSE(360, geoarea.GetComposite(1).GetCornerUR().x, 0.01);
    CHECK_CLOSE(40, geoarea.GetComposite(1).GetCornerUR().y, 0.01);
    CHECK_CLOSE(320, geoarea.GetComposite(1).GetCornerLL().x, 0.01);
    CHECK_CLOSE(30, geoarea.GetComposite(1).GetCornerLL().y, 0.01);
    CHECK_CLOSE(360, geoarea.GetComposite(1).GetCornerLR().x, 0.01);
    CHECK_CLOSE(30, geoarea.GetComposite(1).GetCornerLR().y, 0.01);
}

}
