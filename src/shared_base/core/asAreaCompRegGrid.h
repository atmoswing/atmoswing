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

#ifndef asAreaCompositeRegularGrid_H
#define asAreaCompositeRegularGrid_H

#include <asIncludes.h>
#include <asAreaCompGrid.h>

class asAreaCompRegGrid
        : public asAreaCompGrid
{
public:
    asAreaCompRegGrid(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR, double xStep,
                      double yStep, int flatAllowed = asFLAT_FORBIDDEN, bool isLatLon = true);

    asAreaCompRegGrid(double xMin, double xWidth, double xStep, double yMin, double yWidth, double yStep,
                      int flatAllowed = asFLAT_FORBIDDEN, bool isLatLon = true);

    asAreaCompRegGrid(double xMin, int xPtsNb, double yMin, int yPtsNb, int flatAllowed = asFLAT_FORBIDDEN,
                      bool isLatLon = true);

    ~asAreaCompRegGrid() override = default;

    bool GridsOverlay(asAreaCompGrid *otherArea) const override;

    bool InitializeAxes(const a1d &lons, const a1d &lats, bool strideAllowed = true, bool getLarger = false) override;

    double GetXstep() const override
    {
        return m_xStep;
    }

    double GetYstep() const override
    {
        return m_yStep;
    }

    int GetXstepStride() const
    {
        wxASSERT(m_xStep > 0);
        wxASSERT(fmod(m_xStep, m_xStepData) == 0);
        return static_cast<int>(m_xStep / m_xStepData);
    }

    int GetYstepStride() const
    {
        wxASSERT(m_yStep > 0);
        wxASSERT(fmod(m_yStep, m_yStepData) == 0);
        return static_cast<int>(m_yStep / m_yStepData);
    }

    void SetSameStepAsData()
    {
        m_xStep = m_xStepData;
        m_yStep = m_yStepData;
    }

protected:

private:
    double m_xStep;
    double m_yStep;
    double m_xStepData;
    double m_yStepData;
};

#endif // asAreaCompositeRegularGrid_H
