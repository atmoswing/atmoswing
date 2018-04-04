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

#include "asAreaCompRegGrid.h"

asAreaCompRegGrid::asAreaCompRegGrid(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR,
                                     double xStep, double yStep, int flatAllowed)
        : asAreaCompGrid(cornerUL, cornerUR, cornerLL, cornerLR, flatAllowed),
          m_xStep(xStep),
          m_yStep(yStep)
{
    m_isRegular = true;
}

asAreaCompRegGrid::asAreaCompRegGrid(double xMin, double xWidth, double xStep, double yMin, double yWidth, double yStep,
                                     int flatAllowed)
        : asAreaCompGrid(xMin, xWidth, yMin, yWidth, flatAllowed),
          m_xStep(xStep),
          m_yStep(yStep)
{
    m_isRegular = true;
}

asAreaCompRegGrid::asAreaCompRegGrid(double xMin, int xPtsNb, double yMin, int yPtsNb, int flatAllowed)
        : asAreaCompGrid(xMin, 0, yMin, 0, flatAllowed),
          m_xStep(0),
          m_yStep(0)
{
    m_isRegular = true;
    m_xPtsNb = xPtsNb;
    m_yPtsNb = yPtsNb;
}

bool asAreaCompRegGrid::GridsOverlay(asAreaCompGrid *otherArea) const
{
    if (!otherArea->IsRegular())
        return false;

    auto *otherAreaRegular(dynamic_cast<asAreaCompRegGrid *>(otherArea));

    if (!otherAreaRegular)
        return false;

    if (GetXstep() != otherAreaRegular->GetXstep())
        return false;

    if (GetYstep() != otherAreaRegular->GetYstep())
        return false;

    return true;
}

bool asAreaCompRegGrid::InitializeAxes(const a1d &lons, const a1d &lats, bool strideAllowed)
{
    wxASSERT(lons.size() > 1);
    wxASSERT(lats.size() > 1);

    m_xStepData = abs(lons[1] - lons[0]);
    m_yStepData = abs(lats[1] - lats[0]);

    if (m_xStep == 0) {
        m_xStep = m_xStepData;
    }
    if (m_yStep == 0) {
        m_yStep = m_yStepData;
    }

    if (!asAreaCompGrid::InitializeAxes(lons, lats)) {
        return false;
    }

    if (!strideAllowed) {
        return true;
    }

    if (m_xStep != m_xStepData) {
        if (std::fmod(m_xStep, m_xStepData) != 0) {
            wxLogError(_("Provided step (%d) is not compatible with the data resolution (%d)."), m_xStep, m_xStepData);
            return false;
        }
        auto xFactor = (int) (m_xStep / m_xStepData);

        a1d lonsC0 = m_compositeXaxes[0];
        auto sizeC0new = (int) ((lonsC0.rows() + 1) / xFactor);
        m_compositeXaxes[0].resize(sizeC0new);

        for (int i = 0; i < sizeC0new; ++i) {
            m_compositeXaxes[0][i] = lonsC0[i * xFactor];
        }

        if (GetNbComposites() > 1) {
            double lastXval = m_compositeXaxes[0].tail(1)[0];
            double nextXval = lastXval + m_xStep;

            a1d lonsC1 = m_compositeXaxes[1];

            int indexXmin = asFindClosest(&lonsC1[0], &lonsC1[lonsC1.size() - 1], nextXval, asHIDE_WARNINGS);
            if (indexXmin == asOUT_OF_RANGE) {
                indexXmin = asFindClosest(&lonsC1[0], &lonsC1[lonsC1.size() - 1], nextXval - 360, asHIDE_WARNINGS);
                if (indexXmin == asOUT_OF_RANGE) {
                    indexXmin = asFindClosest(&lonsC1[0], &lonsC1[lonsC1.size() - 1], nextXval + 360);
                }
            }
            wxASSERT(indexXmin >= 0);

            auto sizeC1new = (int) ((lonsC1.rows() - indexXmin + 1) / xFactor);
            m_compositeXaxes[1].resize(sizeC1new);

            for (int i = 0; i < sizeC1new; ++i) {
                m_compositeXaxes[1][i] = lonsC1[indexXmin + i * xFactor];
            }
        }
    }

    if (m_yStep != m_yStepData) {
        if (std::fmod(m_yStep, m_yStepData) != 0) {
            wxLogError(_("Provided step (%d) is not compatible with the data resolution (%d)."), m_yStep, m_yStepData);
            return false;
        }
        auto yFactor = (int) (m_yStep / m_yStepData);

        a1d latsC0 = m_compositeYaxes[0];
        auto sizeC0new = (int) ((latsC0.rows() + 1) / yFactor);
        m_compositeYaxes[0].resize(sizeC0new);

        for (int i = 0; i < sizeC0new; ++i) {
            m_compositeYaxes[0][i] = latsC0[i * yFactor];
        }

        if (GetNbComposites() > 1) {
            m_compositeYaxes[1] = m_compositeYaxes[0];
        }
    }

    return true;
}