/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <iostream>

#include "particle_filter.h"

/// Helper Functions
static double normalize(double angle) {
  angle = fmod(angle, 2.0 * M_PI);
  if (angle > M_PI) {
    angle = 2.0 * M_PI - angle;
  }
  return angle;
}

static LandmarkObs trans(const double &theta,
                         const double &px,
                         const double &py,
                         const double &pn,
                         const double &po,
                         const int &id) {
  return {id, px + std::cos(theta) * pn - std::sin(theta) * po, py + std::sin(theta) * pn + std::cos(theta) * po};
};

///

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 100; //<< 100 particles were enough for me to pass the filter
  particles.resize(num_particles);
  weights.resize(num_particles, 1.0);

  const double stddev_x = std[0];
  const double stddev_y = std[1];
  const double stddev_theta = std[2];

  std::random_device rd;
  std::mt19937 gen_pf(rd());
  std::normal_distribution<double> dist_x(0, stddev_x), dist_y(0, stddev_y), dist_theta(0, stddev_theta);

  for (int i = 0; i < num_particles; ++i) {
    particles[i] = {i, x + dist_x(gen_pf), y + dist_y(gen_pf), normalize(theta + dist_theta(gen_pf)), weights[i]};
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  const double stddev_x = std_pos[0];
  const double stddev_y = std_pos[1];
  const double stddev_theta = std_pos[2];

  std::random_device rd;
  std::mt19937 gen_pf(rd());
  std::normal_distribution<double> dist_x(0, stddev_x), dist_y(0, stddev_y), dist_theta(0, stddev_theta);

  for (int i = 0; i < num_particles; ++i) {
    const double theta_i = particles[i].theta;
    if (fabs(yaw_rate) > 1e-4) {
      particles[i].x += (velocity / yaw_rate) * (std::sin(theta_i + yaw_rate * delta_t) - std::sin(theta_i));
      particles[i].y += (velocity / yaw_rate) * (std::cos(theta_i) - std::cos(theta_i + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
      particles[i].theta = normalize(particles[i].theta);
    } else {
      particles[i].x += velocity * delta_t * std::cos(theta_i);
      particles[i].y += velocity * delta_t * std::sin(theta_i);
      // theta doesn't change
    }

    // some noise
    particles[i].x += dist_x(gen_pf);
    particles[i].y += dist_y(gen_pf);
    particles[i].theta += dist_theta(gen_pf);
    particles[i].theta = normalize(particles[i].theta);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  for (auto &observation : observations) {
    int min_id = -1;
    double min_dist = std::numeric_limits<double>::max();

    for (auto &pred : predicted) {
      double pred_dist = dist(pred.x, pred.y, observation.x, observation.y);
      if (pred_dist < min_dist) {
        min_id = pred.id;
        min_dist = pred_dist;
      }
    }

    observation.id = min_id;

  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html

  for (auto &particle : particles) {
    // landmark association

    std::vector<LandmarkObs> predicted;

    for (auto &map_landmark : map_landmarks.landmark_list) {
      // Consider candidate landmarks close by
      if (fabs(particle.x - map_landmark.x_f) <= sensor_range && fabs(particle.y - map_landmark.y_f) <= sensor_range) {
        predicted.emplace_back(LandmarkObs{map_landmark.id_i, map_landmark.x_f, map_landmark.y_f});
      }
    }

    std::vector<LandmarkObs> observations_to_map;

    for (auto &observation : observations) {
      observations_to_map.emplace_back(trans(particle.theta,
                                             particle.x,
                                             particle.y,
                                             observation.x,
                                             observation.y,
                                             observation.id));
    }

    dataAssociation(predicted, observations_to_map);

    // log weights
    particle.weight = 0.0;

    for (auto &observation_to_map : observations_to_map) {
      for (auto &pred : predicted) {
        if (pred.id == observation_to_map.id) {
          // Log weights for stability
          particle.weight += -std::log(2.0 * M_PI * std_landmark[0] * std_landmark[1])
              - 0.5 * (std::pow((pred.x - observation_to_map.x) / std_landmark[0], 2)
                  + std::pow((pred.y - observation_to_map.y) / std_landmark[1], 2));
        }
      }
    }

    // Inverse to compatible with main.cpp
    particle.weight = std::exp(particle.weight);

  }

}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::vector<double> weights;
  for (auto &particle : particles) {
    weights.emplace_back(particle.weight);
  }

  std::random_device rd;
  std::mt19937 gen_pf(rd());
  std::discrete_distribution<> dist(weights.begin(), weights.end());

  std::vector<Particle> new_particles;
  for (size_t i = 0; i < particles.size(); ++i) {
    new_particles.emplace_back(particles[dist(gen_pf)]);
  }

  std::swap(particles, new_particles);

}

void ParticleFilter::write(std::string filename) {
  // You don't need to modify this file.
  std::ofstream dataFile;
  dataFile.open(filename, std::ios::app);
  for (int i = 0; i < num_particles; ++i) {
    dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
  }
  dataFile.close();
}
