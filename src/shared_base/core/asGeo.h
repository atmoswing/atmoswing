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
 
#ifndef ASGEO_H
#define ASGEO_H

#include <asIncludes.h>

class asGeo: public wxObject
{
public:

    /** Default constructor */
    asGeo();

    /** Default destructor */
    virtual ~asGeo();

    /** Check the point coordinates
     * \param Point The point to check
     */
    bool CheckPoint(Coo &Point, int ChangesAllowed = asEDIT_FORBIDEN);

    double GetAxisXmin()
    {
        return m_axisXmin;
    }

    double GetAxisXmax()
    {
        return m_axisXmax;
    }

    double GetAxisYmin()
    {
        return m_axisYmin;
    }

    double GetAxisYmax()
    {
        return m_axisYmax;
    }

protected:
    double m_axisXmin; //!< Member variable "m_axisXmin"
    double m_axisXmax; //!< Member variable "m_axisXmax"
    double m_axisYmin; //!< Member variable "m_axisYmin"
    double m_axisYmax; //!< Member variable "m_axisYmax"

    /** Initialization */
    void InitBounds();

private:

};

#endif
