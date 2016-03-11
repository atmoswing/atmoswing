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
 
#ifndef ASGEOPOINT_H
#define ASGEOPOINT_H

#include <asIncludes.h>
#include <asGeo.h>

class asGeoPoint: public asGeo
{
public:

    /** Default constructor
     * \param coosys The coordinate system
     * \param Point The coordinates of the point
     * \param Level The height in hPa
     * \param Level The height in m
     */
    asGeoPoint(const Coo &Point, float Level = asNONE, float Height = asNONE);

    /** Other constructor
     * \param coosys The coordinate system
     * \param X The coordinate on the X axis
     * \param Y The coordinate on the Y axis
     * \param Level The height in hPa
     * \param Level The height in m
     */
    asGeoPoint(double x, double y, float Level = asNONE, float Height = asNONE);

    /** Default destructor */
    virtual ~asGeoPoint();

    /** Access m_point
     * \return The current value of m_point
     */
    Coo GetCoo()
    {
        return m_point;
    }

    /** Set m_point
     * \param val New value to set
     */
    void SetCoo(const Coo &val)
    {
        m_point = val;
        Init();
    }

    /** Gives the X coordinate
     * \return The coordinate on the X axis
     */
    double GetX()
    {
        return m_point.x;
    }

    /** Gives the V coordinate
     * \return The coordinate on the Y axis
     */
    double GetY()
    {
        return m_point.y;
    }

    /** Access m_level
     * \return The current value of m_level
     */
    float GetLevel()
    {
        return m_level;
    }

    /** Set m_level
     * \param val New value to set
     */
    void SetLevel(float val)
    {
        m_level = val;
    }

protected:
private:
    Coo m_point; //!< Member variable "m_point"
    float m_level; //!< Member variable "m_level" hPa
    float m_height; //!< Member variable "m_height" m

    /** Process to initialization and checks */
    void Init();

    /** Check every point
     * \return True on success
     */
    bool DoCheckPoints();
};

#endif
