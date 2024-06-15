from FeatureServer import NFServer

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Feature Server')
    parser.add_argument('--config-file', type=str, default='config.yaml')
    args = parser.parse_args()
    server = NFServer(args.config_file)
    server.start()
    server.join()


if __name__ == '__main__':
    main()
